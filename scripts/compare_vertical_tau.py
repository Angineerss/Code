#!/usr/bin/env python3
"""IS CPCV comparison of vertical barrier lengths τ ∈ candidates.

Re-labels existing IS events for each τ on a continuous multi-day bar stream,
applies uniqueness sample weights, and scores RandomForest meta models under
combinatorial purged CV with purge/embargo >= 100 imbalance bars.

Does **not** touch OOS. Requires precomputed daily bars + events under a run
root (warmup+IS only). File layout matches ``scripts/run_range.py``:
``{SYMBOL}_{YYYY-MM-DD}_{bars,events}.csv``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.barriers import apply_triple_barrier
from src.config import PipelineConfig
from src.cpcv import cpcv_splits, seconds_per_imbalance_bar
from src.features import META_FEATURE_NAMES, attach_meta_features
from src.meta_samples import filter_meta_learning_samples
from src.sample_weights import label_uniqueness, sample_weights_from_uniqueness

_DAY_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)_ (?P<day>\d{4}-\d{2}-\d{2})_bars\.csv$".replace(" ", ""))


def _day_paths(run_root: Path, symbol: str) -> list[tuple[date, Path, Path]]:
    out: list[tuple[date, Path, Path]] = []
    for bars_path in sorted(run_root.glob(f"{symbol}_*_bars.csv")):
        m = re.match(rf"^{re.escape(symbol)}_(\d{{4}}-\d{{2}}-\d{{2}})_bars\.csv$", bars_path.name)
        if not m:
            continue
        day = date.fromisoformat(m.group(1))
        events_path = run_root / f"{symbol}_{day.isoformat()}_events.csv"
        if events_path.exists():
            out.append((day, bars_path, events_path))
    return out


def load_continuous_is(
    run_root: Path,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate daily bars/events with remapped bar_ids (IS days only)."""
    config.assert_learning_range(config.universe_start, config.is_end)
    rows = _day_paths(run_root, config.symbol)
    bar_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    offset = 0
    for day, bars_path, events_path in rows:
        if day < config.universe_start or day > config.is_end:
            continue
        bars = pd.read_csv(bars_path)
        events = pd.read_csv(events_path)
        if bars.empty:
            continue
        bars = bars.copy()
        bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
        bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
        bars["bar_id"] = bars["bar_id"].astype(int) + offset
        if not events.empty:
            events = events.copy()
            events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True)
            events["bar_id"] = events["bar_id"].astype(int) + offset
            event_frames.append(events)
        bar_frames.append(bars)
        offset = int(bars["bar_id"].max()) + 1
    if not bar_frames:
        raise SystemExit(f"No IS bars+events under {run_root}")
    bars_all = pd.concat(bar_frames, ignore_index=True)
    events_all = (
        pd.concat(event_frames, ignore_index=True).sort_values("event_ts").reset_index(drop=True)
        if event_frames
        else pd.DataFrame()
    )
    return bars_all, events_all


def relabel_for_tau(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    tau: int,
    config: PipelineConfig,
) -> pd.DataFrame:
    cfg = PipelineConfig(**{**config.__dict__, "vertical_bars": int(tau)})
    # Drop prior barrier columns if present; keep event identity + CUSUM fields.
    drop_cols = [
        c
        for c in (
            "t1_bar_id",
            "t1_ts",
            "touch_type",
            "ret",
            "y_meta",
            "pt_level",
            "sl_level",
            "sigma",
            "flow_strength",
            "tick_rel",
            "label",
            "ret_t1",
            "sample_weight",
            "split",
        )
        if c in events.columns
    ]
    base = events.drop(columns=drop_cols)
    labeled = apply_triple_barrier(bars, base, cfg)
    labeled = attach_meta_features(bars, labeled, vol_span=cfg.barrier_vol_span)
    return labeled


def score_tau(labeled: pd.DataFrame, config: PipelineConfig) -> dict:
    meta = filter_meta_learning_samples(labeled, config)
    need = list(META_FEATURE_NAMES) + ["y_meta"]
    meta = meta.dropna(subset=need).copy()
    if len(meta) < 50:
        return {
            "n": len(meta),
            "error": "too_few_samples",
            "seconds_per_bar": seconds_per_imbalance_bar(meta) if len(meta) else None,
        }

    uniq = label_uniqueness(
        meta["bar_id"].to_numpy(dtype=int),
        meta["t1_bar_id"].to_numpy(dtype=int),
    )
    meta["sample_weight"] = sample_weights_from_uniqueness(uniq)
    X = meta[list(META_FEATURE_NAMES)].to_numpy(dtype=float)
    y = meta["y_meta"].astype(int).to_numpy()
    w = meta["sample_weight"].to_numpy(dtype=float)

    losses: list[float] = []
    accs: list[float] = []
    n_paths = 0
    for tr, te in cpcv_splits(meta, config):
        if len(tr) < 30 or len(te) < 5:
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        clf.fit(X[tr], y[tr], sample_weight=w[tr])
        proba = clf.predict_proba(X[te])
        classes = list(clf.classes_)
        full = np.zeros((len(te), 2), dtype=float)
        for j, c in enumerate(classes):
            full[:, int(c)] = proba[:, j]
        # unseen class column stays 0
        y_te = y[te]
        losses.append(float(log_loss(y_te, full, labels=[0, 1])))
        accs.append(float(accuracy_score(y_te, (full[:, 1] >= 0.5).astype(int))))
        n_paths += 1

    if not losses:
        return {
            "n": int(len(meta)),
            "error": "no_valid_cpcv_paths",
            "seconds_per_bar": seconds_per_imbalance_bar(meta),
            "purge_bars": config.resolved_purge_bars(),
            "embargo_bars": config.resolved_embargo_bars(),
        }
    return {
        "n": int(len(meta)),
        "n_paths": n_paths,
        "logloss_mean": float(np.mean(losses)),
        "logloss_std": float(np.std(losses)),
        "accuracy_mean": float(np.mean(accs)),
        "seconds_per_bar": seconds_per_imbalance_bar(meta),
        "purge_bars": config.resolved_purge_bars(),
        "embargo_bars": config.resolved_embargo_bars(),
        "purge_seconds": int(seconds_per_imbalance_bar(meta) * config.resolved_purge_bars()),
        "embargo_seconds": int(
            seconds_per_imbalance_bar(meta) * config.resolved_embargo_bars()
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument(
        "--taus",
        type=int,
        nargs="+",
        default=None,
        help="Vertical barrier candidates (default: config.vertical_tau_candidates)",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    config = PipelineConfig()
    taus = args.taus or list(config.vertical_tau_candidates)
    print(
        f"τ CPCV compare | taus={taus} | purge={config.resolved_purge_bars()} bars | "
        f"embargo={config.resolved_embargo_bars()} bars | IS only",
        flush=True,
    )
    bars, events = load_continuous_is(args.run_root, config)
    print(f"loaded events={len(events)} bars={len(bars)}", flush=True)

    results: dict[str, dict] = {}
    for tau in taus:
        print(f"\n=== τ={tau} ===", flush=True)
        labeled = relabel_for_tau(events, bars, int(tau), config)
        cfg_tau = PipelineConfig(**{**config.__dict__, "vertical_bars": int(tau)})
        metrics = score_tau(labeled, cfg_tau)
        results[str(tau)] = metrics
        print(json.dumps(metrics, indent=2), flush=True)

    ranked = [(tau, m) for tau, m in results.items() if "logloss_mean" in m]
    ranked.sort(key=lambda x: x[1]["logloss_mean"])
    summary = {
        "purge_bars": config.resolved_purge_bars(),
        "embargo_bars": config.resolved_embargo_bars(),
        "n_bars": int(len(bars)),
        "n_events": int(len(events)),
        "results": results,
        "best_tau": int(ranked[0][0]) if ranked else None,
        "note": "Selected by mean CPCV logloss on IS only; OOS untouched.",
    }
    text = json.dumps(summary, indent=2)
    print("\n=== summary ===", flush=True)
    print(text, flush=True)
    out = args.out or (args.run_root / "vertical_tau_cpcv.json")
    out.write_text(text)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
