#!/usr/bin/env python3
"""Summarize CPCV paths over IS meta-learning samples for a learning run.

Loads daily ``*_labels.csv`` under ``--run-root``, keeps IS-only rows via
``filter_meta_learning_samples`` (policy A: purge+embargo=1τ), and reports
combinatorial purged CV path counts. Refuses any OOS-day artifacts.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig
from src.cpcv import cpcv_splits, seconds_per_imbalance_bar
from src.meta_samples import filter_meta_learning_samples


def _load_is_labels(run_root: Path, config: PipelineConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    oos_hits: list[str] = []
    for path in sorted(run_root.glob(f"{config.symbol}_*_labels.csv")):
        m = re.match(
            rf"^{re.escape(config.symbol)}_(\d{{4}}-\d{{2}}-\d{{2}})_labels\.csv$",
            path.name,
        )
        if not m:
            continue
        day = date.fromisoformat(m.group(1))
        split = config.split_for_day(day)
        if split == "oos":
            oos_hits.append(path.name)
            continue
        if split != "is":
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        frames.append(df)
    if oos_hits:
        raise SystemExit(
            "Refusing CPCV summary: OOS label files present under run-root: "
            + ", ".join(oos_hits[:5])
            + ("..." if len(oos_hits) > 5 else "")
        )
    if not frames:
        raise SystemExit(f"No IS labels under {run_root}")
    out = pd.concat(frames, ignore_index=True)
    out["event_ts"] = pd.to_datetime(out["event_ts"], utc=True, format="mixed")
    out["t1_ts"] = pd.to_datetime(out["t1_ts"], utc=True, format="mixed")
    return out.sort_values("event_ts").reset_index(drop=True)


def main() -> None:
    config = PipelineConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    config.assert_learning_range(config.universe_start, config.is_end)
    labeled = _load_is_labels(args.run_root, config)
    meta = filter_meta_learning_samples(labeled, config)

    paths = list(cpcv_splits(meta, config))
    train_sizes = [int(len(tr)) for tr, _ in paths]
    test_sizes = [int(len(te)) for _, te in paths]
    spb = float(seconds_per_imbalance_bar(meta)) if len(meta) else None

    summary = {
        "run_root": str(args.run_root),
        "n_is_labels_raw": int(len(labeled)),
        "n_meta_after_boundary": int(len(meta)),
        "y_meta_rate": None if meta.empty else float(meta["y_meta"].mean()),
        "cv_method": config.cv_method,
        "selection_method": config.selection_method,
        "n_cpcv_groups": config.n_cpcv_groups,
        "n_cpcv_test_groups": config.n_cpcv_test_groups,
        "expected_paths": config.cpcv_path_count(),
        "n_paths": len(paths),
        "train_size_mean": float(np.mean(train_sizes)) if train_sizes else None,
        "test_size_mean": float(np.mean(test_sizes)) if test_sizes else None,
        "vertical_bars": config.vertical_bars,
        "purge_bars": config.resolved_purge_bars(),
        "embargo_bars": config.resolved_embargo_bars(),
        "seconds_per_bar": spb,
        "purge_seconds": None if spb is None else int(spb * config.resolved_purge_bars()),
        "embargo_seconds": None
        if spb is None
        else int(spb * config.resolved_embargo_bars()),
        "oos_range": [config.oos_start.isoformat(), config.oos_end.isoformat()],
        "oos_touched": False,
        "boundary_policy": "A_purge_embargo_1tau",
    }
    text = json.dumps(summary, indent=2)
    print(text, flush=True)
    out = args.out or (args.run_root / "is_cpcv_summary.json")
    out.write_text(text + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
