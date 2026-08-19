"""Screen dollar-bar clocks by bar shape (Method B). No labels, no CPCV, no OOS.

Used to lock ``imbalance_divisor=650``: among 400 vs 650, pick the faster
clock. See ``results/divisor_bar_screen.json`` and
``scripts/compare_divisor_bars.py``.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

NEAR_ZERO = 0.05
NEAR_ONE = 0.95
SHORT_S = 5.0
LONG_S = 4.0 * 3600.0
CLIP_EPS = 0.02


def usable_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or bars.empty:
        return bars.iloc[0:0] if bars is not None else pd.DataFrame()
    if "close_reason" not in bars.columns:
        return bars
    return bars.loc[bars["close_reason"] != "warmup"].reset_index(drop=True)


def flow_strength(bars: pd.DataFrame) -> np.ndarray:
    if bars.empty:
        return np.array([], dtype=float)
    flow = np.abs(bars["signed_flow"].to_numpy(dtype=float))
    if "quote_volume" in bars.columns:
        denom = np.abs(bars["quote_volume"].to_numpy(dtype=float))
    else:
        denom = np.abs(bars["threshold"].to_numpy(dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return flow / np.maximum(denom, 1e-12)


def _pct(values: np.ndarray, qs: tuple[int, ...] = (10, 50, 90, 99)) -> dict[str, float | None]:
    if values.size == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def duration_stats(duration_s: np.ndarray) -> dict:
    out = _pct(duration_s)
    if duration_s.size == 0:
        out.update(share_shorter_than_5s=None, share_longer_than_4h=None, n=0)
        return out
    out.update(
        share_shorter_than_5s=float((duration_s < SHORT_S).mean()),
        share_longer_than_4h=float((duration_s > LONG_S).mean()),
        n=int(duration_s.size),
    )
    return out


def strength_stats(strength: np.ndarray) -> dict:
    out = _pct(strength)
    if strength.size == 0:
        out.update(share_near_0=None, share_near_1=None, n=0)
        return out
    finite = strength[np.isfinite(strength)]
    out.update(
        share_near_0=float((finite < NEAR_ZERO).mean()) if finite.size else None,
        share_near_1=float((finite > NEAR_ONE).mean()) if finite.size else None,
        n=int(finite.size),
    )
    return out


def clip_stats(dollar_threshold: np.ndarray, d_seed: float) -> dict:
    if not np.isfinite(d_seed) or d_seed <= 0 or dollar_threshold.size == 0:
        return {"share_at_0_5d": None, "share_at_2d": None, "median_t_over_d": None, "n": 0}
    ratio = dollar_threshold / d_seed
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        return {"share_at_0_5d": None, "share_at_2d": None, "median_t_over_d": None, "n": 0}
    lo = 0.5 + CLIP_EPS
    hi = 2.0 - CLIP_EPS
    return {
        "share_at_0_5d": float((finite <= lo).mean()),
        "share_at_2d": float((finite >= hi).mean()),
        "median_t_over_d": float(np.median(finite)),
        "n": int(finite.size),
    }


def quiet_loud_stats(daily: pd.DataFrame) -> dict:
    """Compare bar counts on quiet vs busy days (bottom/top 20% by day quote)."""
    need = {"n_bars", "daily_quote"}
    if daily.empty or not need.issubset(daily.columns) or len(daily) < 10:
        return {
            "n_days": int(len(daily)),
            "quiet_median_bars": None,
            "loud_median_bars": None,
            "loud_over_quiet_bars": None,
            "loud_over_quiet_quote": None,
        }
    q = daily["daily_quote"].to_numpy(dtype=float)
    lo = float(np.nanpercentile(q, 20))
    hi = float(np.nanpercentile(q, 80))
    quiet = daily.loc[daily["daily_quote"] <= lo]
    loud = daily.loc[daily["daily_quote"] >= hi]
    q_bars = float(quiet["n_bars"].median()) if not quiet.empty else None
    l_bars = float(loud["n_bars"].median()) if not loud.empty else None
    q_quote = float(quiet["daily_quote"].median()) if not quiet.empty else None
    l_quote = float(loud["daily_quote"].median()) if not loud.empty else None
    bar_ratio = None if not q_bars else (None if q_bars == 0 else l_bars / q_bars)
    quote_ratio = None if not q_quote else (None if q_quote == 0 else l_quote / q_quote)
    return {
        "n_days": int(len(daily)),
        "n_quiet_days": int(len(quiet)),
        "n_loud_days": int(len(loud)),
        "quiet_median_bars": q_bars,
        "loud_median_bars": l_bars,
        "loud_over_quiet_bars": None if bar_ratio is None else float(bar_ratio),
        "loud_over_quiet_quote": None if quote_ratio is None else float(quote_ratio),
        "median_bars": float(daily["n_bars"].median()),
        "cv_bars": float(daily["n_bars"].std(ddof=0) / daily["n_bars"].mean())
        if float(daily["n_bars"].mean()) > 0
        else None,
        "cv_quote": float(daily["daily_quote"].std(ddof=0) / daily["daily_quote"].mean())
        if float(daily["daily_quote"].mean()) > 0
        else None,
    }


def day_row(
    day: date,
    window: str,
    divisor: int,
    bars: pd.DataFrame,
    daily_quote: float,
    d_seed: float,
) -> dict:
    use = usable_bars(bars)
    strength = flow_strength(use)
    duration = (
        use["duration_s"].to_numpy(dtype=float)
        if not use.empty and "duration_s" in use.columns
        else np.array([], dtype=float)
    )
    thr = (
        use["dollar_threshold"].to_numpy(dtype=float)
        if not use.empty and "dollar_threshold" in use.columns
        else np.array([], dtype=float)
    )
    reasons = (
        use["close_reason"].value_counts().to_dict()
        if not use.empty and "close_reason" in use.columns
        else {}
    )
    n_max = int(reasons.get("max_ticks", 0))
    return {
        "day": day.isoformat(),
        "window": window,
        "divisor": int(divisor),
        "n_bars": int(len(use)),
        "daily_quote": float(daily_quote) if np.isfinite(daily_quote) else None,
        "D": float(d_seed) if np.isfinite(d_seed) else None,
        "median_duration_s": None if duration.size == 0 else float(np.median(duration)),
        "median_flow_strength": None if strength.size == 0 else float(np.median(strength)),
        "share_max_ticks": None if len(use) == 0 else n_max / len(use),
        "n_max_ticks": n_max,
        **{f"duration_{k}": v for k, v in duration_stats(duration).items() if k != "n"},
        **{f"strength_{k}": v for k, v in strength_stats(strength).items() if k != "n"},
        **{f"clip_{k}": v for k, v in clip_stats(thr, d_seed).items() if k != "n"},
        "_duration": duration,
        "_strength": strength,
        "_threshold": thr,
    }


def summarize_divisor(daily: pd.DataFrame, duration: np.ndarray, strength: np.ndarray) -> dict:
    scored = daily.drop(columns=[c for c in daily.columns if c.startswith("_")], errors="ignore")
    d_seed = scored["D"].to_numpy(dtype=float) if "D" in scored else np.array([])
    # clip stats need per-bar T$; fall back to daily medians if arrays empty
    pooled_clip = clip_stats(np.array([], dtype=float), float("nan"))
    if "clip_median_t_over_d" in scored.columns:
        pooled_clip = {
            "share_at_0_5d": float(scored["clip_share_at_0_5d"].mean())
            if scored["clip_share_at_0_5d"].notna().any()
            else None,
            "share_at_2d": float(scored["clip_share_at_2d"].mean())
            if scored["clip_share_at_2d"].notna().any()
            else None,
            "median_t_over_d": float(scored["clip_median_t_over_d"].median())
            if scored["clip_median_t_over_d"].notna().any()
            else None,
        }
    reasons_max = (
        float(scored["share_max_ticks"].mean())
        if "share_max_ticks" in scored and scored["share_max_ticks"].notna().any()
        else None
    )
    return {
        "divisor": None if scored.empty else int(scored["divisor"].iloc[0]),
        "n_scored_days": int(len(scored)),
        "median_bars_per_day": None if scored.empty else float(scored["n_bars"].median()),
        "duration": duration_stats(duration),
        "flow_strength": strength_stats(strength),
        "clip": pooled_clip,
        "share_max_ticks": reasons_max,
        "quiet_loud": quiet_loud_stats(scored),
        "mean_D": None if d_seed.size == 0 else float(np.nanmean(d_seed)),
    }
