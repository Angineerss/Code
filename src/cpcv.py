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


def _purge_train(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    embargo: int,
) -> np.ndarray:
    """Drop train events whose label interval overlaps a test event, plus embargo."""
    if train_idx.size == 0 or test_idx.size == 0:
        return train_idx
    test_start = t0[test_idx].min()
    test_end = t1[test_idx].max()
    embargo_end = test_end + np.timedelta64(embargo, "s") if embargo else test_end
    keep = []
    for i in train_idx:
        overlaps = not (t1[i] < test_start or t0[i] > embargo_end)
        if not overlaps:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def cpcv_splits(
    labeled: pd.DataFrame,
    config: PipelineConfig,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) paths with purge + embargo.

    Purge/embargo *ratios* stay TBD; until then lengths default to the
    vertical barrier (``config.resolved_*_bars``). Embargo is approximated
    as ``embargo_bars * median bar duration``.
    """
    n = len(labeled)
    if n == 0:
        return
    groups = _contiguous_groups(n, config.n_cpcv_groups)
    if len(groups) < 2:
        return

    t0 = labeled["event_ts"].to_numpy()
    t1 = labeled["t1_ts"].to_numpy()
    bar_dt = pd.Series(t1).diff().median()
    if pd.isna(bar_dt):
        bar_dt = pd.Timedelta(seconds=0)
    embargo_seconds = int(max(bar_dt.total_seconds(), 0) * config.resolved_embargo_bars())

    k = min(config.n_cpcv_test_groups, len(groups) - 1)
    for test_combo in combinations(range(len(groups)), k):
        test_idx = np.concatenate([groups[i] for i in test_combo])
        train_parts = [groups[i] for i in range(len(groups)) if i not in test_combo]
        if not train_parts:
            continue
        train_idx = np.concatenate(train_parts)
        train_idx = _purge_train(train_idx, test_idx, t0, t1, embargo_seconds)
        yield np.sort(train_idx), np.sort(test_idx)
