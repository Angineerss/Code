"""Combinatorial purged cross-validation over labeled events."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def _contiguous_groups(n: int, n_groups: int) -> list[np.ndarray]:
    edges = np.linspace(0, n, n_groups + 1, dtype=int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(n_groups) if edges[i + 1] > edges[i]]


def seconds_per_imbalance_bar(labeled: pd.DataFrame) -> float:
    """Median seconds per imbalance bar from labeled paths (or event spacing)."""
    if labeled.empty:
        return 60.0
    t0 = pd.to_datetime(labeled["event_ts"], utc=True)
    t1 = pd.to_datetime(labeled["t1_ts"], utc=True)
    if {"bar_id", "t1_bar_id"}.issubset(labeled.columns):
        n_bars = (
            pd.to_numeric(labeled["t1_bar_id"], errors="coerce")
            - pd.to_numeric(labeled["bar_id"], errors="coerce")
        ).to_numpy(dtype=float)
        secs = (t1 - t0).dt.total_seconds().to_numpy(dtype=float)
        ok = np.isfinite(n_bars) & np.isfinite(secs) & (n_bars > 0) & (secs > 0)
        if ok.any():
            return float(np.median(secs[ok] / n_bars[ok]))
    ts = t0.sort_values()
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return 60.0
    return float(diffs.median().total_seconds())


def _purge_train(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    purge_seconds: int,
    embargo_seconds: int,
) -> np.ndarray:
    """Drop train events whose label interval overlaps the purged test neighborhood.

    Purge expands *before* the test window by ``purge_seconds`` (bar-count based).
    Embargo expands *after* the test window by ``embargo_seconds``.
    """
    if train_idx.size == 0 or test_idx.size == 0:
        return train_idx
    test_start = t0[test_idx].min()
    test_end = t1[test_idx].max()
    purge_start = test_start - np.timedelta64(int(purge_seconds), "s")
    embargo_end = test_end + np.timedelta64(int(embargo_seconds), "s")
    keep = []
    for i in train_idx:
        overlaps = not (t1[i] < purge_start or t0[i] > embargo_end)
        if not overlaps:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def cpcv_splits(
    labeled: pd.DataFrame,
    config: PipelineConfig,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) paths with purge + embargo.

    Runs on **IS-labeled events only**. Contiguous time groups form the CV
    partition: each path uses ``n_cpcv_test_groups`` as the CV fold and the
    remaining groups as train (after purge/embargo). OOS must not enter here.

    Purge/embargo lengths are ``resolved_*_bars() * median imbalance-bar duration``.
    Locked policy A: each defaults to 1τ (``vertical_bars``) so train does not
    bleed into nearby evaluation windows (and, at the IS/OOS cut, into OOS).
    """
    n = len(labeled)
    if n == 0:
        return
    groups = _contiguous_groups(n, config.n_cpcv_groups)
    if len(groups) < 2:
        return

    t0 = labeled["event_ts"].to_numpy()
    t1 = labeled["t1_ts"].to_numpy()
    spb = seconds_per_imbalance_bar(labeled)
    purge_seconds = int(max(spb * config.resolved_purge_bars(), 0.0))
    embargo_seconds = int(max(spb * config.resolved_embargo_bars(), 0.0))

    k = min(config.n_cpcv_test_groups, len(groups) - 1)
    for test_combo in combinations(range(len(groups)), k):
        test_idx = np.concatenate([groups[i] for i in test_combo])
        train_parts = [groups[i] for i in range(len(groups)) if i not in test_combo]
        if not train_parts:
            continue
        train_idx = np.concatenate(train_parts)
        train_idx = _purge_train(
            train_idx, test_idx, t0, t1, purge_seconds, embargo_seconds
        )
        yield np.sort(train_idx), np.sort(test_idx)
