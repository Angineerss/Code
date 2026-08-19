"""Download one UTC day of Binance aggTrades from data.binance.vision."""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

VISION_BASE = "https://data.binance.vision/data"

AGGTRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "qty",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
]


def aggtrades_url(symbol: str, day: date, market: str = "spot") -> str:
    fname = f"{symbol.upper()}-aggTrades-{day.isoformat()}.zip"
    return f"{VISION_BASE}/{market}/daily/aggTrades/{symbol.upper()}/{fname}"


def monthly_aggtrades_url(symbol: str, year: int, month: int, market: str = "spot") -> str:
    ym = f"{year:04d}-{month:02d}"
    fname = f"{symbol.upper()}-aggTrades-{ym}.zip"
    return f"{VISION_BASE}/{market}/monthly/aggTrades/{symbol.upper()}/{fname}"


def months_from_to(start: date, end: date) -> list[tuple[int, int]]:
    """Inclusive calendar months that overlap [start, end]."""
    months: list[tuple[int, int]] = []
    year, month = start.year, start.month
    end_ym = (end.year, end.month)
    while (year, month) <= end_ym:
        months.append((year, month))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return months


def _download_to(path: Path, url: str, timeout: int = 600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    req = Request(url, headers={"User-Agent": "structlabel/0.1"})
    with urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(path)
    return path


def download_aggtrades_month(
    symbol: str,
    year: int,
    month: int,
    dest_dir: Path,
    market: str = "spot",
    timeout: int = 1800,
    skip_existing: bool = True,
) -> Path | None:
    zip_path = dest_dir / f"{symbol.upper()}-aggTrades-{year:04d}-{month:02d}.zip"
    if skip_existing and zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path
    url = monthly_aggtrades_url(symbol, year, month, market)
    if not _http_ok(url + ".CHECKSUM") and not _http_ok(url):
        return None
    return _download_to(zip_path, url, timeout=timeout)


def _http_ok(url: str, timeout: int = 30) -> bool:
    for method in ("HEAD", "GET"):
        req = Request(url, method=method, headers={"User-Agent": "structlabel/0.1"})
        try:
            with urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                return 200 <= status < 300
        except HTTPError as exc:
            if exc.code == 405 and method == "HEAD":
                continue
            return False
        except (URLError, TimeoutError, ValueError):
            return False
    return False


def latest_available_day(
    symbol: str,
    market: str = "spot",
    lookback_days: int = 10,
    as_of: date | None = None,
) -> date:
    """Pick the most recent UTC day that Binance Vision already published."""
    start = as_of or datetime.now(timezone.utc).date() - timedelta(days=1)
    for i in range(lookback_days):
        day = start - timedelta(days=i)
        url = aggtrades_url(symbol, day, market)
        if _http_ok(url + ".CHECKSUM") or _http_ok(url):
            return day
    raise FileNotFoundError(
        f"No published {symbol} aggTrades in the last {lookback_days} days "
        f"before {start.isoformat()}"
    )


def _decode_timestamp_ms(raw: np.ndarray) -> np.ndarray:
    """Binance files mix ms and us epoch units depending on era."""
    values = raw.astype(np.int64)
    if values.size and np.nanmedian(values) > 10**14:
        return values // 1000
    return values


_CSV_CHUNKSIZE = 1_000_000


def _empty_ticks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": pd.Series(dtype="int64"),
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "price": pd.Series(dtype="float64"),
            "qty": pd.Series(dtype="float64"),
            "side": pd.Series(dtype="int8"),
            "quote_qty": pd.Series(dtype="float64"),
        }
    )


def _finalize_aggtrades(df: pd.DataFrame) -> pd.DataFrame:
    ts_ms = _decode_timestamp_ms(df["timestamp"].to_numpy())
    maker = df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"])
    side = np.where(maker, -1, 1)
    out = pd.DataFrame(
        {
            "trade_id": df["agg_trade_id"].astype(np.int64),
            "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
            "price": pd.to_numeric(df["price"], errors="coerce"),
            "qty": pd.to_numeric(df["qty"], errors="coerce"),
            "side": side.astype(np.int8),
        }
    )
    out["quote_qty"] = out["price"] * out["qty"]
    return out.dropna(subset=["price", "qty", "timestamp"])


def _aggtrades_reader(raw: bytes | io.BufferedReader):
    fh = io.BytesIO(raw) if isinstance(raw, (bytes, bytearray)) else raw
    return pd.read_csv(
        fh,
        header=None,
        names=AGGTRADE_COLUMNS[:7],
        usecols=range(7),
        chunksize=_CSV_CHUNKSIZE,
    )


def parse_aggtrades_csv(raw: bytes | io.BufferedReader) -> pd.DataFrame:
    parts = [_finalize_aggtrades(chunk) for chunk in _aggtrades_reader(raw)]
    if not parts:
        return _empty_ticks()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("timestamp").reset_index(drop=True)


def _concat_tick_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return _empty_ticks()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("timestamp").reset_index(drop=True)


def _iter_zip_aggtrade_frames(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV inside {zip_path}")
        with zf.open(names[0]) as fh:
            for chunk in _aggtrades_reader(fh):
                yield _finalize_aggtrades(chunk)


def download_aggtrades_day(
    symbol: str,
    day: date,
    dest_dir: Path,
    market: str = "spot",
    timeout: int = 120,
    skip_existing: bool = True,
) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{symbol.upper()}-aggTrades-{day.isoformat()}.zip"
    if skip_existing and zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path
    url = aggtrades_url(symbol, day, market)
    if not _http_ok(url + ".CHECKSUM") and not _http_ok(url):
        return None
    return _download_to(zip_path, url, timeout=timeout)


def load_aggtrades_zip(zip_path: Path) -> pd.DataFrame:
    return _concat_tick_parts(list(_iter_zip_aggtrade_frames(zip_path)))


def load_monthly_ticks_by_day(zip_path: Path) -> dict[date, pd.DataFrame]:
    """Split a monthly Vision zip into UTC days without holding the raw CSV twice."""
    buckets: dict[date, list[pd.DataFrame]] = {}
    for frame in _iter_zip_aggtrade_frames(zip_path):
        if frame.empty:
            continue
        day_col = frame["timestamp"].dt.tz_convert("UTC").dt.date
        for day_key, part in frame.groupby(day_col, sort=False):
            buckets.setdefault(day_key, []).append(part)
    return {day_key: _concat_tick_parts(parts) for day_key, parts in buckets.items()}


def filter_ticks_day(ticks: pd.DataFrame, day: date) -> pd.DataFrame:
    start = pd.Timestamp(day, tz="UTC")
    end = start + pd.Timedelta(days=1)
    out = ticks.loc[(ticks["timestamp"] >= start) & (ticks["timestamp"] < end)]
    return out.reset_index(drop=True)


def monthly_archive_path(archive_dir: Path, symbol: str, day: date) -> Path:
    return archive_dir / "monthly" / f"{symbol.upper()}-aggTrades-{day.year:04d}-{day.month:02d}.zip"


def daily_archive_path(archive_dir: Path, symbol: str, day: date) -> Path:
    return archive_dir / "daily" / f"{symbol.upper()}-aggTrades-{day.isoformat()}.zip"


def load_day_from_archive(
    symbol: str,
    day: date,
    archive_dir: Path,
    month_cache: dict[tuple[int, int], dict[date, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Load one UTC day from local Vision archive (daily zip preferred, else monthly)."""
    daily = daily_archive_path(archive_dir, symbol, day)
    if daily.exists() and daily.stat().st_size > 0:
        return load_aggtrades_zip(daily)

    monthly = monthly_archive_path(archive_dir, symbol, day)
    if not monthly.exists() or monthly.stat().st_size <= 0:
        raise FileNotFoundError(
            f"No archive ticks for {symbol} on {day}: missing {daily.name} and {monthly.name}"
        )
    key = (day.year, day.month)
    if month_cache is not None:
        if key not in month_cache:
            month_cache[key] = load_monthly_ticks_by_day(monthly)
        by_day = month_cache[key]
        if day not in by_day:
            raise FileNotFoundError(f"Archive month {monthly.name} has no rows for {day}")
        day_ticks = by_day.pop(day)
        if not by_day:
            del month_cache[key]
        return day_ticks

    start = pd.Timestamp(day, tz="UTC")
    end = start + pd.Timedelta(days=1)
    parts = []
    for frame in _iter_zip_aggtrade_frames(monthly):
        part = frame.loc[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]
        if not part.empty:
            parts.append(part)
    day_ticks = _concat_tick_parts(parts)
    if day_ticks.empty:
        raise FileNotFoundError(f"Archive month {monthly.name} has no rows for {day}")
    return day_ticks


def load_or_download_day(
    symbol: str,
    dest_dir: Path,
    market: str = "spot",
    day: date | None = None,
    archive_dir: Path | None = None,
) -> tuple[pd.DataFrame, date]:
    chosen = day or latest_available_day(symbol, market)
    if archive_dir is not None:
        return load_day_from_archive(symbol, chosen, archive_dir), chosen
    zip_path = dest_dir / f"{symbol.upper()}-aggTrades-{chosen.isoformat()}.zip"
    if not zip_path.exists():
        path = download_aggtrades_day(symbol, chosen, dest_dir, market, skip_existing=False)
        if path is None:
            raise FileNotFoundError(f"No aggTrades for {symbol} on {chosen}")
    return load_aggtrades_zip(zip_path), chosen


def download_aggtrades_archive(
    symbol: str,
    dest_dir: Path,
    start: date,
    end: date,
    market: str = "spot",
) -> dict:
    """Download monthly zips for completed months and daily zips for the final open month.

    Completed months: [start, end] months whose last day is <= end and the month is
    fully closed relative to ``end`` (month < end's month, or end is month-end).
    Remaining days in the final month are fetched as daily files.
    """
    monthly_dir = dest_dir / "monthly"
    daily_dir = dest_dir / "daily"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)

    months = months_from_to(start, end)
    downloaded_months: list[str] = []
    skipped_months: list[str] = []
    missing_months: list[str] = []
    downloaded_days: list[str] = []
    missing_days: list[str] = []

    for year, month in months:
        if month == 12:
            month_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = date(year, month + 1, 1) - timedelta(days=1)
        # Prefer monthly zip when the whole month is at or before ``end``.
        use_monthly = month_end <= end
        label = f"{year:04d}-{month:02d}"
        if use_monthly:
            before = monthly_dir / f"{symbol.upper()}-aggTrades-{label}.zip"
            existed = before.exists() and before.stat().st_size > 0
            print(f"[monthly] {label} ...", flush=True)
            path = download_aggtrades_month(symbol, year, month, monthly_dir, market)
            if path is None:
                missing_months.append(label)
                print(f"[monthly] {label} MISSING", flush=True)
            elif existed:
                skipped_months.append(label)
                print(f"[monthly] {label} skip ({path.stat().st_size} bytes)", flush=True)
            else:
                downloaded_months.append(label)
                print(f"[monthly] {label} ok ({path.stat().st_size} bytes)", flush=True)
            continue
        # Partial final month: daily files from max(start, month_start) through end.
        day = max(start, date(year, month, 1))
        while day <= end:
            before = daily_dir / f"{symbol.upper()}-aggTrades-{day.isoformat()}.zip"
            existed = before.exists() and before.stat().st_size > 0
            print(f"[daily] {day.isoformat()} ...", flush=True)
            path = download_aggtrades_day(symbol, day, daily_dir, market)
            if path is None:
                missing_days.append(day.isoformat())
                print(f"[daily] {day.isoformat()} MISSING", flush=True)
            elif not existed:
                downloaded_days.append(day.isoformat())
                print(f"[daily] {day.isoformat()} ok ({path.stat().st_size} bytes)", flush=True)
            else:
                print(f"[daily] {day.isoformat()} skip", flush=True)
            day += timedelta(days=1)

    return {
        "symbol": symbol.upper(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "monthly_dir": str(monthly_dir),
        "daily_dir": str(daily_dir),
        "downloaded_months": downloaded_months,
        "skipped_months": skipped_months,
        "missing_months": missing_months,
        "downloaded_days": downloaded_days,
        "missing_days": missing_days,
    }
