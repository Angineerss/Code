"""Prior-year daily quote notional for the imbalance-bar threshold."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .download import VISION_BASE

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "n_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


@dataclass(frozen=True)
class PriorYearNotional:
    start: date
    end: date
    n_days: int
    average_daily_notional: float
    average_daily_trades: float | None
    expected_size: float | None
    init_t: int | None
    threshold: float


def prior_year_window(as_of: date, lookback_days: int = 365) -> tuple[date, date]:
    """Sliding UTC window ending yesterday: [as_of - lookback, as_of - 1d]."""
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    return as_of - timedelta(days=lookback_days), as_of - timedelta(days=1)


def threshold_from_average(average_daily_notional: float, divisor: int = 650) -> float:
    if average_daily_notional <= 0:
        raise ValueError("average daily notional must be positive")
    return float(average_daily_notional) / float(divisor)


def months_covering(start: date, end: date) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return months


def parse_klines(payload: list) -> pd.DataFrame:
    rows = [
        {
            "open_time": pd.to_datetime(int(item[0]), unit="ms", utc=True),
            "quote_volume": float(item[7]),
            "n_trades": float(item[8]),
        }
        for item in payload
    ]
    return pd.DataFrame(rows)


def parse_klines_csv(raw: bytes | io.BufferedReader) -> pd.DataFrame:
    df = pd.read_csv(raw, header=None, names=KLINE_COLUMNS)
    open_time = df["open_time"].astype("int64")
    if len(open_time) and open_time.median() > 10**14:
        open_time = open_time // 1000
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(open_time, unit="ms", utc=True),
            "quote_volume": pd.to_numeric(df["quote_volume"], errors="coerce"),
            "n_trades": pd.to_numeric(df["n_trades"], errors="coerce"),
        }
    ).dropna(subset=["quote_volume"])


def monthly_klines_url(symbol: str, year: int, month: int, market: str = "spot") -> str:
    ym = f"{year:04d}-{month:02d}"
    fname = f"{symbol.upper()}-1d-{ym}.zip"
    return f"{VISION_BASE}/{market}/monthly/klines/{symbol.upper()}/1d/{fname}"


def _download_bytes(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": "structlabel/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_monthly_klines(symbol: str, year: int, month: int, cache_dir: Path | None = None) -> pd.DataFrame:
    zip_name = f"{symbol.upper()}-1d-{year:04d}-{month:02d}.zip"
    zip_path = None if cache_dir is None else cache_dir / zip_name
    if zip_path is not None and zip_path.exists():
        raw = zip_path.read_bytes()
    else:
        raw = _download_bytes(monthly_klines_url(symbol, year, month))
        if zip_path is not None:
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV in {zip_name}")
        with zf.open(names[0]) as fh:
            return parse_klines_csv(fh)


def fetch_daily_quote_notional_api(symbol: str, start: date, end: date) -> pd.DataFrame:
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_exclusive = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)
    end_ms = int(end_exclusive.timestamp() * 1000) - 1
    url = (
        f"{BINANCE_KLINES}?symbol={symbol.upper()}&interval=1d"
        f"&startTime={start_ms}&endTime={end_ms}&limit=1000"
    )
    payload = json.loads(_download_bytes(url).decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        raise FileNotFoundError(f"No 1d klines for {symbol} between {start} and {end}")
    return parse_klines(payload)


def fetch_daily_quote_notional(
    symbol: str,
    start: date,
    end: date,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    frames = [fetch_monthly_klines(symbol, year, month, cache_dir) for year, month in months_covering(start, end)]
    daily = pd.concat(frames, ignore_index=True)
    days = daily["open_time"].dt.tz_convert("UTC").dt.date
    return daily.loc[(days >= start) & (days <= end)].reset_index(drop=True)


def load_or_fetch_daily_quote_notional(
    symbol: str,
    start: date,
    end: date,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{symbol.upper()}_1d_{start.isoformat()}_{end.isoformat()}.csv"
        if cache_path.exists():
            df = pd.read_csv(cache_path, parse_dates=["open_time"])
            if not df.empty and "quote_volume" in df.columns:
                df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
                return df
    try:
        df = fetch_daily_quote_notional(symbol, start, end, cache_dir)
    except (HTTPError, URLError, TimeoutError, ValueError):
        df = fetch_daily_quote_notional_api(symbol, start, end)
    if cache_dir is not None:
        df.to_csv(cache_path, index=False)
    return df


def prior_year_notional(
    symbol: str,
    as_of: date,
    divisor: int = 650,
    lookback_days: int = 365,
    cache_dir: Path | None = None,
) -> PriorYearNotional:
    start, end = prior_year_window(as_of, lookback_days)
    daily = load_or_fetch_daily_quote_notional(symbol, start, end, cache_dir)
    if daily.empty:
        raise FileNotFoundError(f"No daily quote volume for {symbol} in {start}..{end}")
    average = float(daily["quote_volume"].mean())
    avg_trades = None
    expected_size = None
    init_t = None
    if "n_trades" in daily and daily["n_trades"].notna().any():
        avg_trades = float(daily["n_trades"].mean())
        if avg_trades > 0:
            expected_size = average / avg_trades
            init_t = max(int(round(avg_trades / divisor)), 1)
    return PriorYearNotional(
        start=start,
        end=end,
        n_days=int(len(daily)),
        average_daily_notional=average,
        average_daily_trades=avg_trades,
        expected_size=expected_size,
        init_t=init_t,
        threshold=threshold_from_average(average, divisor),
    )
