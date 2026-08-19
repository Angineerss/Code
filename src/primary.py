"""Primary model: direction of the bet, applied after the event filter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig, PrimaryType


def apply_primary(bars: pd.DataFrame, events: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Attach primary ``side`` from the dollar-imbalance formula, then gates.

    Default primary is ``sign(signed_flow)`` = sign(θ) on the dollar bar. [선정]
    Weak bars stay in: AFML triple-barrier primary maximizes recall.
    ``require_strong_imbalance`` is contrast-only.
    """
    if events.empty:
        out = events.copy()
        out["side"] = pd.Series(dtype="int8")
        return out

    side = _primary_side(bars, events, config.primary_type)
    out = events.copy()
    out["side"] = side.astype(np.int8)
    out = out.loc[out["side"] != 0]
    if "close_reason" in bars.columns:
        reason = out["bar_id"].map(bars.set_index("bar_id")["close_reason"])
        out = out.loc[reason != "warmup"]
    if config.require_cusum_flow_agree and "cusum_side" in out.columns:
        cusum = pd.to_numeric(out["cusum_side"], errors="coerce")
        if cusum.notna().any():
            out = out.loc[cusum.notna() & (cusum == out["side"])]
    return filter_strong_imbalance(bars, out, config).reset_index(drop=True)


def filter_strong_imbalance(
    bars: pd.DataFrame, events: pd.DataFrame, config: PipelineConfig
) -> pd.DataFrame:
    """Keep events where |θ| ≥ E[θ] (dollar-imbalance formula). Contrast-only."""
    if events.empty or not config.require_strong_imbalance:
        return events
    by_id = bars.set_index("bar_id")
    flow = pd.to_numeric(events["bar_id"].map(by_id["signed_flow"]), errors="coerce")
    thr = pd.to_numeric(events["bar_id"].map(by_id["threshold"]), errors="coerce")
    keep = flow.abs() >= thr.abs().clip(lower=1e-12)
    return events.loc[keep]


def _primary_side(bars: pd.DataFrame, events: pd.DataFrame, primary_type: PrimaryType) -> np.ndarray:
    if primary_type == "rule_bar_flow_sign":
        if "signed_flow" not in bars.columns:
            raise ValueError("rule_bar_flow_sign requires bars['signed_flow']")
        flow = bars.set_index("bar_id")["signed_flow"]
        mapped = events["bar_id"].map(flow).to_numpy(dtype=float)
        return np.sign(mapped)
    if primary_type == "rule_cusum_sign":
        if "cusum_side" not in events.columns:
            raise ValueError("rule_cusum_sign requires CUSUM event times")
        return events["cusum_side"].to_numpy(dtype=np.int8)
    raise ValueError(f"Unknown primary_type: {primary_type}")
