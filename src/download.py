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
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{symbol.upper()}-aggTrades-{day.isoformat()}.zip"
    url = aggtrades_url(symbol, day, market)
    req = Request(url, headers={"User-Agent": "structlabel/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        zip_path.write_bytes(resp.read())
    return zip_path


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
        download_aggtrades_day(symbol, chosen, dest_dir, market)
    return load_aggtrades_zip(zip_path), chosen
