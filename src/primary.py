"""Primary model: direction of the bet, applied after the event filter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig, PrimaryType


def apply_primary(bars: pd.DataFrame, events: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Attach primary ``side`` to CUSUM (or every-bar) event times.

    CUSUM decides *when*; primary decides *which way* (default = bar flow sign).
    With ``require_cusum_flow_agree``, keep only events where ``cusum_side == side``
    so taker imbalance and the price-run direction align (locked hypothesis).
    """
    if events.empty:
        out = events.copy()
        out["side"] = pd.Series(dtype="int8")
        return out

    side = _primary_side(bars, events, config.primary_type)
    out = events.copy()
    out["side"] = side.astype(np.int8)
    out = out.loc[out["side"] != 0]
    if config.require_cusum_flow_agree and "cusum_side" in out.columns:
        cusum = pd.to_numeric(out["cusum_side"], errors="coerce")
        # every_bar mode has no CUSUM side — skip the agree gate.
        if cusum.notna().any():
            out = out.loc[cusum.notna() & (cusum == out["side"])]
    return out.reset_index(drop=True)


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
