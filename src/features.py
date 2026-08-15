"""Meta-label features at event time (no look-ahead).

Hypothesis strength:
- flow_strength: |θ| / E[θ] on the event bar
- cusum_excess_ratio: |S| / h at the CUSUM crossing (before reset)

Context (locked subset):
- tick_rel: tick_count / E[T] at bar close (E[T] before EWMA update)
- sigma: same EWM bar-log-return vol used for barriers
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .barriers import barrier_volatility

META_FEATURE_NAMES = (
    "flow_strength",
    "cusum_excess_ratio",
    "tick_rel",
    "sigma",
)


def attach_meta_features(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    vol_span: int = 50,
) -> pd.DataFrame:
    """Append locked meta features to labeled or unlabeled events."""
    out = events.copy()
    if out.empty:
        for name in META_FEATURE_NAMES:
            out[name] = pd.Series(dtype="float64")
        return out

    by_id = bars.set_index("bar_id")
    flow = out["bar_id"].map(by_id["signed_flow"]).to_numpy(dtype=float)
    bar_thr = out["bar_id"].map(by_id["threshold"]).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        strength = np.abs(flow) / np.maximum(np.abs(bar_thr), 1e-12)
    out["flow_strength"] = strength

    if "cusum_excess_ratio" in out.columns:
        out["cusum_excess_ratio"] = pd.to_numeric(out["cusum_excess_ratio"], errors="coerce")
    else:
        out["cusum_excess_ratio"] = np.nan

    ticks = out["bar_id"].map(by_id["tick_count"]).to_numpy(dtype=float)
    et = out["bar_id"].map(by_id["expected_ticks"]).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["tick_rel"] = ticks / np.maximum(et, 1e-12)

    # Prefer barrier-computed sigma on labeled rows; otherwise match barrier formula.
    if "sigma" in out.columns and pd.to_numeric(out["sigma"], errors="coerce").notna().all():
        out["sigma"] = pd.to_numeric(out["sigma"], errors="coerce")
    else:
        sig = barrier_volatility(bars, vol_span)
        sig_by_id = pd.Series(sig, index=bars["bar_id"].to_numpy())
        out["sigma"] = out["bar_id"].map(sig_by_id).astype(float)
    return out


def meta_feature_matrix(labeled: pd.DataFrame) -> pd.DataFrame:
    """Return X = locked feature columns only."""
    missing = [c for c in META_FEATURE_NAMES if c not in labeled.columns]
    if missing:
        raise KeyError(f"Missing meta features: {missing}")
    return labeled.loc[:, list(META_FEATURE_NAMES)].astype(float)
