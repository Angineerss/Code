"""Symmetric CUSUM event filter on imbalance-bar log prices (AFML).

CUSUM only selects *when* to consider a bet. Direction comes from the primary model.
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
    """Use each imbalance-bar close as an event time (no primary side yet)."""
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
    """
    if bars.empty:
        return pd.DataFrame(columns=["bar_id", "event_ts", "threshold", "cusum_side"])

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
        level = h[i] if np.isfinite(h[i]) else h[i - 1]
        if s_neg < -level:
            rows.append([int(bar_id[i]), ts[i], float(level), -1])
            s_neg = 0.0
        elif s_pos > level:
            rows.append([int(bar_id[i]), ts[i], float(level), 1])
            s_pos = 0.0

    events = pd.DataFrame(rows, columns=["bar_id", "event_ts", "threshold", "cusum_side"])
    if not events.empty:
        events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True)
        events["cusum_side"] = events["cusum_side"].astype(np.int8)
    return events
