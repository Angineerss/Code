"""Filters for meta-learning samples (AFML hygiene).

Learning uses IS events only:
- warmup excluded (EWMA/D seed only; incomplete 365d prior)
- OOS untouched until final evaluation
- IS↔OOS boundary purge (+ optional embargo)
- hyperparams / features / MDA / meta threshold chosen only via CPCV on IS
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .config import PipelineConfig


def _event_day(ts) -> date:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.date()


def add_split_column(labeled: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Attach ``split`` from each row's event calendar day."""
    out = labeled.copy()
    if out.empty:
        out["split"] = pd.Series(dtype="object")
        return out
    days = out["event_ts"].map(_event_day)
    out["split"] = [config.split_for_day(d) for d in days]
    return out


def boundary_purge_mask(labeled: pd.DataFrame, config: PipelineConfig) -> pd.Series:
    """True = keep. Drop rows whose label end ``t1_ts`` reaches OOS."""
    if labeled.empty:
        return pd.Series(dtype=bool)
    oos_start = pd.Timestamp(config.oos_start, tz="UTC")
    return pd.to_datetime(labeled["t1_ts"], utc=True) < oos_start


def boundary_embargo_mask(labeled: pd.DataFrame, config: PipelineConfig) -> pd.Series:
    """True = keep. Drop IS events whose ``event_ts`` falls inside the pre-OOS embargo.

    Embargo length defaults to one max label lifetime (``resolved_embargo_bars`` /
    ``vertical_bars`` × median observed label horizon), matching AFML's use of
    the holding-period scale at a purged boundary.
    """
    if labeled.empty:
        return pd.Series(dtype=bool)
    event_ts = pd.to_datetime(labeled["event_ts"], utc=True)
    t1_ts = pd.to_datetime(labeled["t1_ts"], utc=True)
    horizons = t1_ts - event_ts
    finite = horizons[horizons.notna() & (horizons > pd.Timedelta(0))]
    if finite.empty:
        med = pd.Timedelta(days=0)
    else:
        med = finite.median()
    scale = float(config.resolved_embargo_bars()) / float(max(config.vertical_bars, 1))
    embargo = med * scale
    cutoff = pd.Timestamp(config.oos_start, tz="UTC") - embargo
    return event_ts < cutoff


def filter_meta_learning_samples(
    labeled: pd.DataFrame,
    config: PipelineConfig,
    *,
    apply_boundary_purge: bool | None = None,
    apply_boundary_embargo: bool | None = None,
) -> pd.DataFrame:
    """IS-only meta-training rows: no warmup, no OOS, optional boundary purge/embargo."""
    if labeled.empty:
        return labeled.copy()
    out = add_split_column(labeled, config)
    out = out.loc[out["split"] == "is"].copy()
    purge = config.boundary_purge if apply_boundary_purge is None else apply_boundary_purge
    embargo = config.boundary_embargo if apply_boundary_embargo is None else apply_boundary_embargo
    if purge and not out.empty:
        out = out.loc[boundary_purge_mask(out, config)].copy()
    if embargo and not out.empty:
        out = out.loc[boundary_embargo_mask(out, config)].copy()
    return out.reset_index(drop=True)


def assert_cpcv_only_selection(config: PipelineConfig) -> None:
    if config.selection_method != "cpcv_only":
        raise ValueError(
            "Hyperparameters, features, MDA, and meta thresholds must be selected "
            f"via CPCV only (selection_method={config.selection_method!r})"
        )
