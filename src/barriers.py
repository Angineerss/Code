"""Triple-barrier path labeling and meta-labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .cusum import ewm_std


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except (TypeError, ValueError):
        return None
    return int(value)


def barrier_volatility(bars: pd.DataFrame, span: int) -> np.ndarray:
    vol = ewm_std(bars["log_ret"].to_numpy(dtype=float), span)
    finite = vol[np.isfinite(vol)]
    fill = float(np.median(finite)) if finite.size else 0.001
    return np.where(np.isfinite(vol), vol, fill)


def apply_triple_barrier(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Label each event by first touch of TP / SL / vertical barrier.

    ``events.side`` is the primary bet. Meta target ``y_meta`` is 1 only on
    take-profit. Stop-loss, timeout, and simultaneous touches are 0.
    """
    if events.empty or bars.empty:
        return events.assign(
            t1_bar_id=pd.Series(dtype="int64"),
            t1_ts=pd.Series(dtype="datetime64[ns, UTC]"),
            touch_type=pd.Series(dtype="object"),
            ret=pd.Series(dtype="float64"),
            y_meta=pd.Series(dtype="int8"),
            pt_level=pd.Series(dtype="float64"),
            sl_level=pd.Series(dtype="float64"),
            sigma=pd.Series(dtype="float64"),
        )

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    end_ts = bars["end_ts"].to_numpy()
    bar_id = bars["bar_id"].to_numpy()
    id_to_pos = {int(b): i for i, b in enumerate(bar_id)}
    sigma = barrier_volatility(bars, config.barrier_vol_span)

    records: list[dict[str, object]] = []
    n = len(bars)
    for ev in events.itertuples(index=False):
        pos = id_to_pos[int(ev.bar_id)]
        side = int(ev.side)
        entry = close[pos]
        sig = float(sigma[pos])
        pt_level = entry * (1.0 + side * config.pt * sig)
        sl_level = entry * (1.0 - side * config.sl * sig)
        last = min(pos + config.vertical_bars, n - 1)
        touch_type = "timeout"
        exit_pos = last
        for j in range(pos + 1, last + 1):
            hit_tp = high[j] >= pt_level if side > 0 else low[j] <= pt_level
            hit_sl = low[j] <= sl_level if side > 0 else high[j] >= sl_level
            if hit_tp and hit_sl:
                touch_type = "simultaneous"
                exit_pos = j
                break
            if hit_tp:
                touch_type = "take_profit"
                exit_pos = j
                break
            if hit_sl:
                touch_type = "stop_loss"
                exit_pos = j
                break
        ret = side * (close[exit_pos] / entry - 1.0)
        if touch_type == "take_profit":
            y_meta = 1
        elif touch_type == "simultaneous":
            y_meta = int(config.simultaneous_touch_y)
        else:
            y_meta = int(config.timeout_y) if touch_type == "timeout" else 0
        records.append(
            {
                "bar_id": int(ev.bar_id),
                "event_ts": ev.event_ts,
                "side": side,
                "cusum_side": _optional_int(getattr(ev, "cusum_side", None)),
                "threshold": float(ev.threshold),
                "t1_bar_id": int(bar_id[exit_pos]),
                "t1_ts": end_ts[exit_pos],
                "touch_type": touch_type,
                "ret": float(ret),
                "y_meta": y_meta,
                "pt_level": float(pt_level),
                "sl_level": float(sl_level),
                "sigma": sig,
            }
        )

    labeled = pd.DataFrame.from_records(records)
    labeled["event_ts"] = pd.to_datetime(labeled["event_ts"], utc=True)
    labeled["t1_ts"] = pd.to_datetime(labeled["t1_ts"], utc=True)
    labeled["side"] = labeled["side"].astype(np.int8)
    labeled["y_meta"] = labeled["y_meta"].astype(np.int8)
    return labeled
