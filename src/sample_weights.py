"""Sample uniqueness weights and MDA feature importance for meta labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def label_uniqueness(bar_id: np.ndarray, t1_bar_id: np.ndarray) -> np.ndarray:
    """AFML-style average uniqueness over concurrent [bar_id, t1_bar_id] intervals."""
    starts = bar_id.astype(int)
    ends = t1_bar_id.astype(int)
    if len(starts) == 0:
        return np.asarray([], dtype=float)
    conc = np.zeros(int(ends.max()) + 1, dtype=float)
    for s, e in zip(starts, ends):
        if e < s:
            e = s
        conc[s : e + 1] += 1.0
    uniq = np.empty(len(starts), dtype=float)
    for i, (s, e) in enumerate(zip(starts, ends)):
        if e < s:
            e = s
        c = conc[s : e + 1]
        uniq[i] = float(np.mean(1.0 / np.maximum(c, 1.0)))
    return uniq


def sample_weights_from_uniqueness(uniq: np.ndarray) -> np.ndarray:
    """Normalize uniqueness to mean 1 for RF sample_weight."""
    if len(uniq) == 0:
        return uniq
    mu = float(np.mean(uniq))
    if mu <= 0:
        return np.ones_like(uniq, dtype=float)
    return uniq / mu


def mda_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    n_splits: int = 5,
    n_estimators: int = 200,
    random_state: int = 0,
) -> pd.DataFrame:
    """Mean-decrease-accuracy: shuffle one feature at a time, measure score drop.

    Uses KFold on rows (diagnostic). For production IS use CPCV paths instead.
    """
    features = list(X.columns)
    x = X.to_numpy(dtype=float)
    y = np.asarray(y)
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    kf = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=random_state)
    drops = {f: [] for f in features}
    base_scores: list[float] = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(x)):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=max(2, len(train_idx) // 50),
            random_state=random_state + fold_i,
            n_jobs=-1,
            class_weight=None,
        )
        clf.fit(x[train_idx], y[train_idx], sample_weight=sample_weight[train_idx])
        pred = clf.predict(x[test_idx])
        # weighted accuracy on test
        w_te = sample_weight[test_idx]
        base = float(np.average(pred == y[test_idx], weights=w_te))
        base_scores.append(base)
        rng = np.random.default_rng(random_state + 100 + fold_i)
        for j, name in enumerate(features):
            x_perm = x[test_idx].copy()
            x_perm[:, j] = rng.permutation(x_perm[:, j])
            pred_p = clf.predict(x_perm)
            score_p = float(np.average(pred_p == y[test_idx], weights=w_te))
            drops[name].append(base - score_p)

    rows = []
    for name in features:
        arr = np.asarray(drops[name], dtype=float)
        rows.append(
            {
                "feature": name,
                "mda_mean": float(arr.mean()),
                "mda_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "mda_folds": len(arr),
            }
        )
    out = pd.DataFrame(rows).sort_values("mda_mean", ascending=False).reset_index(drop=True)
    out.attrs["base_accuracy_mean"] = float(np.mean(base_scores))
    return out
