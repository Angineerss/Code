"""Dollar-bar closes are the default event times.

``event_mode='every_bar'`` keeps every close so the primary can maximize recall.
CUSUM remains available as ``event_mode='cusum'`` (contrast only).
``bar_type='dollar_imbalance'`` is the original clock (control).
Direction comes from the primary model (sign of bar θ).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .primary import apply_primary


def ewm_std(values: np.ndarray, span: int) -> np.ndarray:
    s = pd.Series(values, dtype=float)
    return s.ewm(span=span, min_periods=max(span // 2, 2), adjust=False).std().to_numpy()


def cusum_threshold(log_ret: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Volatility-scaled CUSUM barrier: h_t = k * σ_t.

    Default k=1 uses one EWM standard deviation of the bar log-return series.
    """
    if config.cusum_mode == "absolute":
        return np.full(len(log_ret), config.cusum_absolute_h, dtype=float)
    vol = ewm_std(log_ret, config.cusum_vol_span)
    vol = np.where(np.isfinite(vol), vol, np.nan)
    # Fall back to a small constant until the EWMA window fills.
    fill = np.nanmedian(vol[np.isfinite(vol)]) if np.isfinite(vol).any() else config.cusum_absolute_h
    vol = np.where(np.isfinite(vol), vol, fill)
    return config.cusum_k * vol


def every_bar_events(bars: pd.DataFrame) -> pd.DataFrame:
    """Use each bar close as an event time (no primary side yet)."""
    if bars.empty:
        return pd.DataFrame(columns=["bar_id", "event_ts", "threshold"])
    return pd.DataFrame(
        {
            "bar_id": bars["bar_id"].to_numpy(),
            "event_ts": pd.to_datetime(bars["end_ts"], utc=True),
            "threshold": bars["threshold"].to_numpy(dtype=float),
        }
    )


def select_events(bars: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    if config.event_mode == "every_bar":
        times = every_bar_events(bars)
    else:
        times = cusum_events(bars, config)
    return apply_primary(bars, times, config)


def cusum_events(bars: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """AFML symmetric CUSUM on bar log-returns (snippet 2.4).

    S+_t = max(0, S+_{t-1} + y_t), S-_t = min(0, S-_{t-1} + y_t),
    with y_t = Δ log price. Only the side that crosses ±h is reset to 0.
    ``cusum_side`` is the crossing that triggered the filter, not the primary bet.
    ``cusum_excess_ratio`` = |S|/h at the crossing (before reset).
    """
    empty_cols = ["bar_id", "event_ts", "threshold", "cusum_side", "cusum_excess_ratio"]
    if bars.empty:
        return pd.DataFrame(columns=empty_cols)

    close = bars["close"].to_numpy(dtype=float)
    log_p = np.log(close)
    delta = np.diff(log_p, prepend=log_p[0])
    h = cusum_threshold(bars["log_ret"].to_numpy(dtype=float), config)

    s_pos = 0.0
    s_neg = 0.0
    rows: list[list[object]] = []
    ts = bars["end_ts"].to_numpy()
    bar_id = bars["bar_id"].to_numpy()

    for i in range(1, len(bars)):
        s_pos = max(0.0, s_pos + delta[i])
        s_neg = min(0.0, s_neg + delta[i])
        level = float(h[i] if np.isfinite(h[i]) else h[i - 1])
        level = max(level, 1e-12)
        if s_neg < -level:
            rows.append([int(bar_id[i]), ts[i], level, -1, float((-s_neg) / level)])
            s_neg = 0.0
        elif s_pos > level:
            rows.append([int(bar_id[i]), ts[i], level, 1, float(s_pos / level)])
            s_pos = 0.0

    events = pd.DataFrame(rows, columns=empty_cols)
    if not events.empty:
        events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True)
        events["cusum_side"] = events["cusum_side"].astype(np.int8)
    return events
