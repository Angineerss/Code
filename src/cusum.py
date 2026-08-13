"""Symmetric CUSUM filter on imbalance-bar log prices (AFML)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def ewm_std(values: np.ndarray, span: int) -> np.ndarray:
    s = pd.Series(values, dtype=float)
    return s.ewm(span=span, min_periods=max(span // 2, 2), adjust=False).std().to_numpy()


def cusum_threshold(log_ret: np.ndarray, config: PipelineConfig) -> np.ndarray:
    if config.cusum_mode == "absolute":
        return np.full(len(log_ret), config.cusum_absolute_h, dtype=float)
    vol = ewm_std(log_ret, config.cusum_vol_span)
    vol = np.where(np.isfinite(vol), vol, np.nan)
    # Fall back to a small constant until the EWMA window fills.
    fill = np.nanmedian(vol[np.isfinite(vol)]) if np.isfinite(vol).any() else config.cusum_absolute_h
    vol = np.where(np.isfinite(vol), vol, fill)
    return config.cusum_k * vol


def every_bar_events(bars: pd.DataFrame) -> pd.DataFrame:
    """Use each imbalance-bar close as an event; side = order-flow sign."""
    if bars.empty:
        return pd.DataFrame(columns=["bar_id", "event_ts", "side", "threshold"])
    side = np.sign(bars["signed_flow"].to_numpy(dtype=float)).astype(np.int8)
    out = pd.DataFrame(
        {
            "bar_id": bars["bar_id"].to_numpy(),
            "event_ts": pd.to_datetime(bars["end_ts"], utc=True),
            "side": side,
            "threshold": bars["threshold"].to_numpy(dtype=float),
        }
    )
    return out.loc[out["side"] != 0].reset_index(drop=True)


def select_events(bars: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    if config.event_mode == "every_bar":
        return every_bar_events(bars)
    return cusum_events(bars, config)


def cusum_events(bars: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Emit an event when cumulative log-price drift exceeds h.

    ``side`` is the CUSUM direction: +1 up-cross, -1 down-cross.
    This is the high-recall rule-based primary signal.
    """
    if bars.empty:
        return pd.DataFrame(columns=["bar_id", "event_ts", "side", "threshold"])

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
        if s_pos > level:
            rows.append([int(bar_id[i]), ts[i], 1, float(level)])
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg < -level:
            rows.append([int(bar_id[i]), ts[i], -1, float(level)])
            s_pos = 0.0
            s_neg = 0.0

    events = pd.DataFrame(rows, columns=["bar_id", "event_ts", "side", "threshold"])
    if not events.empty:
        events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True)
        events["side"] = events["side"].astype(np.int8)
    return events
