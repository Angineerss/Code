#!/usr/bin/env python3
"""Structure bars over the learning range only.

Default clock is dollar bars (treatment). Pass ``--bar-type dollar_imbalance``
for the original control sampler. Use a separate ``--out-dir`` (the default
path includes bar_type) so EWMA files are never mixed.

Split rules (locked):
- Warmup ``archive_start..warmup_end``: bars + EWMA only (D seed / state).
  No events, primary, or meta labels.
- IS ``universe_start..is_end``: bars → events → primary (sign of θ) →
  triple-barrier → meta features. CV inside IS is **CPCV only**.
- OOS ``oos_start..oos_end``: never touched here (no ``--allow-oos``).

This is a thin orchestrator over ``scripts/run_range.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    config = PipelineConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default=config.symbol)
    p.add_argument("--archive-dir", default="data/aggtrades")
    p.add_argument("--klines-dir", default="data/klines")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Default: data/runs/learning_{bar_type}_{archive_start}_{is_end}",
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--warmup-only",
        action="store_true",
        help="Stop after warmup bars/EWMA (no IS labels).",
    )
    p.add_argument(
        "--is-only",
        action="store_true",
        help="Skip warmup; require prior-day EWMA already in out-dir.",
    )
    p.add_argument(
        "--primary",
        default=config.primary_type,
        choices=("rule_bar_flow_sign", "rule_cusum_sign"),
    )
    p.add_argument(
        "--bar-type",
        default=config.bar_type,
        choices=("dollar", "tick_imbalance", "volume_imbalance", "dollar_imbalance"),
        help="dollar = treatment. dollar_imbalance = original control sampler.",
    )
    args = p.parse_args(argv)

    out_dir = Path(
        args.out_dir
        or f"data/runs/learning_{args.bar_type}_{config.archive_start}_{config.is_end}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hard guard: learning range must not intersect OOS.
    config.assert_learning_range(config.archive_start, config.is_end)

    manifest = {
        "symbol": args.symbol.upper(),
        "bar_type": args.bar_type,
        "clock_role": (
            "treatment"
            if args.bar_type == "dollar"
            else "control"
            if args.bar_type == "dollar_imbalance"
            else "other"
        ),
        "out_dir": str(out_dir),
        "warmup": [config.archive_start.isoformat(), config.warmup_end.isoformat()],
        "is": [config.universe_start.isoformat(), config.is_end.isoformat()],
        "oos_locked": [config.oos_start.isoformat(), config.oos_end.isoformat()],
        "cv_method": config.cv_method,
        "selection_method": config.selection_method,
        "n_cpcv_groups": config.n_cpcv_groups,
        "n_cpcv_test_groups": config.n_cpcv_test_groups,
        "cpcv_paths": config.cpcv_path_count(),
        "purge_bars": config.resolved_purge_bars(),
        "embargo_bars": config.resolved_embargo_bars(),
        "vertical_bars": config.vertical_bars,
        "require_strong_imbalance": config.require_strong_imbalance,
        "boundary_policy": "A_purge_embargo_1tau",
        "note": "OOS excluded. IS labels validated only via CPCV.",
    }
    (out_dir / "learning_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)

    py = sys.executable
    range_script = str(ROOT / "scripts" / "run_range.py")
    common = [
        py,
        range_script,
        "--symbol",
        args.symbol.upper(),
        "--archive-dir",
        args.archive_dir,
        "--klines-dir",
        args.klines_dir,
        "--out-dir",
        str(out_dir),
        "--primary",
        args.primary,
        "--bar-type",
        args.bar_type,
    ]
    if args.skip_existing:
        common.append("--skip-existing")

    if not args.is_only:
        # Warmup: bars + EWMA only.
        rc = _run(
            common
            + [
                "--start",
                config.archive_start.isoformat(),
                "--end",
                config.warmup_end.isoformat(),
                "--bars-only",
            ]
        )
        if rc != 0:
            return rc
        if args.warmup_only:
            print(json.dumps({"done": True, "phase": "warmup"}, indent=2), flush=True)
            return 0

    # IS: full labeling. Still no OOS (run_range asserts learning range).
    rc = _run(
        common
        + [
            "--start",
            config.universe_start.isoformat(),
            "--end",
            config.is_end.isoformat(),
        ]
    )
    if rc != 0:
        return rc

    print(
        json.dumps(
            {
                "done": True,
                "phase": "is",
                "next": (
                    f"python scripts/summarize_is_cpcv.py --run-root {out_dir} "
                    "(CPCV on IS meta samples; OOS still untouched)"
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
