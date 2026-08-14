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
    with urlopen(req, timeout=timeout) as resp:
        tmp.write_bytes(resp.read())
    tmp.replace(path)
    return path


def download_aggtrades_month(
    symbol: str,
    year: int,
    month: int,
    dest_dir: Path,
    market: str = "spot",
    timeout: int = 600,
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


def parse_aggtrades_csv(raw: bytes | io.BufferedReader) -> pd.DataFrame:
    df = pd.read_csv(
        raw,
        header=None,
        names=AGGTRADE_COLUMNS[:7],
        usecols=range(7),
    )
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
    out = out.dropna(subset=["price", "qty", "timestamp"]).sort_values("timestamp")
    return out.reset_index(drop=True)


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
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV inside {zip_path}")
        with zf.open(names[0]) as fh:
            return parse_aggtrades_csv(fh)


def load_or_download_day(
    symbol: str,
    dest_dir: Path,
    market: str = "spot",
    day: date | None = None,
) -> tuple[pd.DataFrame, date]:
    chosen = day or latest_available_day(symbol, market)
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
            path = download_aggtrades_month(symbol, year, month, monthly_dir, market)
            if path is None:
                missing_months.append(label)
            elif existed:
                skipped_months.append(label)
            else:
                downloaded_months.append(label)
            continue
        # Partial final month: daily files from max(start, month_start) through end.
        day = max(start, date(year, month, 1))
        while day <= end:
            before = daily_dir / f"{symbol.upper()}-aggTrades-{day.isoformat()}.zip"
            existed = before.exists() and before.stat().st_size > 0
            path = download_aggtrades_day(symbol, day, daily_dir, market)
            if path is None:
                missing_days.append(day.isoformat())
            elif not existed:
                downloaded_days.append(day.isoformat())
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
