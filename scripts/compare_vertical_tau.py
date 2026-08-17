#!/usr/bin/env python3
"""IS CPCV comparison of vertical barrier lengths τ ∈ candidates.

Re-labels existing IS events for each τ on a continuous multi-day bar stream,
applies uniqueness sample weights, and scores RandomForest meta models under
combinatorial purged CV with purge/embargo = 1τ bars (policy A).

Does **not** touch OOS. Requires precomputed daily bars + events under a run
root (warmup+IS only). File layout matches ``scripts/run_range.py``:
``{SYMBOL}_{YYYY-MM-DD}_{bars,events}.csv``.

Scores whatever clock is already in ``--run-root``. Treatment vs control:
run ``run_learning_range.py`` twice (default dollar, then
``--bar-type dollar_imbalance``) into separate folders, then point this
script at each root. Do not mix bar types in one folder.
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
    start: date | None = None,
    end: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Concatenate daily bars/events with remapped bar_ids (IS days only)."""
    lo = start or config.universe_start
    hi = end or config.is_end
    if lo < config.universe_start or hi > config.is_end:
        raise SystemExit("τ compare window must stay inside IS (OOS locked)")
    config.assert_learning_range(lo, hi)
    rows = _day_paths(run_root, config.symbol)
    bar_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    days: list[date] = []
    offset = 0
    for day, bars_path, events_path in rows:
        if day < lo or day > hi:
            continue
        bars = pd.read_csv(bars_path)
        events = pd.read_csv(events_path)
        if bars.empty:
            continue
        bars = bars.copy()
        bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True, format="mixed")
        bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True, format="mixed")
        bars["bar_id"] = bars["bar_id"].astype(int) + offset
        if not events.empty:
            events = events.copy()
            events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True, format="mixed")
            events["bar_id"] = events["bar_id"].astype(int) + offset
            event_frames.append(events)
        bar_frames.append(bars)
        days.append(day)
        offset = int(bars["bar_id"].max()) + 1
    if not bar_frames:
        raise SystemExit(f"No IS bars+events under {run_root}")
    bars_all = pd.concat(bar_frames, ignore_index=True)
    events_all = (
        pd.concat(event_frames, ignore_index=True).sort_values("event_ts").reset_index(drop=True)
        if event_frames
        else pd.DataFrame()
    )
    window = {
        "first_day": days[0].isoformat(),
        "last_day": days[-1].isoformat(),
        "n_days": len(days),
        "requested_start": lo.isoformat(),
        "requested_end": hi.isoformat(),
        "is_full_is": lo == config.universe_start and hi == config.is_end and len(days)
        == (config.is_end - config.universe_start).days + 1,
    }
    return bars_all, events_all, window


def _full_horizon_mask(events: pd.DataFrame, bars: pd.DataFrame, tau: int) -> pd.Series:
    """True when the bar stream still has ``tau`` bars after the event bar."""
    id_to_pos = {int(b): i for i, b in enumerate(bars["bar_id"].to_numpy())}
    n = len(bars)
    pos = events["bar_id"].map(id_to_pos)
    return (pos + int(tau)) <= (n - 1)


def relabel_for_tau(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    tau: int,
    config: PipelineConfig,
    pt: float | None = None,
    sl: float | None = None,
) -> pd.DataFrame:
    overrides: dict = {**config.__dict__, "vertical_bars": int(tau)}
    if pt is not None:
        overrides["pt"] = float(pt)
    if sl is not None:
        overrides["sl"] = float(sl)
    cfg = PipelineConfig(**overrides)
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
    keep = _full_horizon_mask(base, bars, int(tau))
    n_dropped = int((~keep).sum())
    base = base.loc[keep].reset_index(drop=True)
    labeled = apply_triple_barrier(bars, base, cfg)
    labeled = attach_meta_features(bars, labeled, vol_span=cfg.barrier_vol_span)
    labeled.attrs["n_dropped_horizon"] = n_dropped
    return labeled


def score_tau(labeled: pd.DataFrame, config: PipelineConfig) -> dict:
    meta = filter_meta_learning_samples(labeled, config)
    feat_names = [
        c
        for c in META_FEATURE_NAMES
        if c in meta.columns and pd.to_numeric(meta[c], errors="coerce").notna().any()
    ]
    need = feat_names + ["y_meta"]
    if not feat_names:
        return {"n": int(len(meta)), "error": "no_meta_features"}
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
    X = meta[feat_names].to_numpy(dtype=float)
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

    touch = labeled["touch_type"].value_counts().to_dict() if "touch_type" in labeled else {}
    y_rate = None if labeled.empty or "y_meta" not in labeled else float(labeled["y_meta"].mean())
    entropy = None
    if y_rate is not None and 0.0 < y_rate < 1.0:
        entropy = float(-(y_rate * np.log(y_rate) + (1.0 - y_rate) * np.log(1.0 - y_rate)))
    held = None
    if not labeled.empty and {"bar_id", "t1_bar_id"}.issubset(labeled.columns):
        held = (
            pd.to_numeric(labeled["t1_bar_id"], errors="coerce")
            - pd.to_numeric(labeled["bar_id"], errors="coerce")
        )
    extras = {
        "n_labeled": int(len(labeled)),
        "n_dropped_horizon": int(labeled.attrs.get("n_dropped_horizon", 0)),
        "y_meta_rate": y_rate,
        "label_entropy": entropy,
        "touch_types": {str(k): int(v) for k, v in touch.items()},
        "timeout_rate": float(touch.get("timeout", 0) / len(labeled)) if len(labeled) else None,
        "median_bars_to_touch": None if held is None or held.empty else float(held.median()),
        "p90_bars_to_touch": None if held is None or held.empty else float(held.quantile(0.9)),
        "vertical_bars": config.vertical_bars,
        "pt": config.pt,
        "sl": config.sl,
        "meta_features": feat_names,
    }
    if not losses:
        return {
            "n": int(len(meta)),
            "error": "no_valid_cpcv_paths",
            "seconds_per_bar": seconds_per_imbalance_bar(meta),
            "purge_bars": config.resolved_purge_bars(),
            "embargo_bars": config.resolved_embargo_bars(),
            **extras,
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
        **extras,
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
    p.add_argument("--start", default=None, help="IS window start YYYY-MM-DD (inclusive)")
    p.add_argument("--end", default=None, help="IS window end YYYY-MM-DD (inclusive)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    config = PipelineConfig()
    taus = args.taus or list(config.vertical_tau_candidates)
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    print(
        f"τ CPCV compare | taus={taus} | purge/embargo = 1τ bars (policy A) | IS only",
        flush=True,
    )
    bars, events, window = load_continuous_is(args.run_root, config, start=start, end=end)
    print(
        f"loaded events={len(events)} bars={len(bars)} "
        f"days={window['n_days']} {window['first_day']}..{window['last_day']} "
        f"full_is={window['is_full_is']}",
        flush=True,
    )

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
        "window": f"{window['first_day']}..{window['last_day']}",
        "n_days": window["n_days"],
        "is_full_is": window["is_full_is"],
        "split": "is",
        "oos_touched": False,
        "selection_metric": "mean_cpcv_logloss",
        "boundary_policy": "A_purge_embargo_1tau",
        "n_cpcv_groups": config.n_cpcv_groups,
        "n_cpcv_test_groups": config.n_cpcv_test_groups,
        "cpcv_paths": config.cpcv_path_count(),
        "n_bars": int(len(bars)),
        "n_events": int(len(events)),
        "results": results,
        "best_tau": int(ranked[0][0]) if ranked else None,
        "note": (
            "Selected by mean CPCV logloss on available IS events only; OOS untouched. "
            "Each τ uses purge/embargo = 1τ. Events without a full τ horizon at the "
            "end of the loaded bar stream are dropped. Partial IS is not a production lock."
            if not window["is_full_is"]
            else "Selected by mean CPCV logloss on IS only; OOS untouched. "
            "Each τ uses purge/embargo = 1τ."
        ),
    }
    text = json.dumps(summary, indent=2)
    print("\n=== summary ===", flush=True)
    print(text, flush=True)
    out = args.out or (args.run_root / "vertical_tau_cpcv.json")
    out.write_text(text)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
