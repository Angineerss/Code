#!/usr/bin/env python3
"""IS CPCV comparison of symmetric pt=sl widths, optionally crossed with τ.

Re-labels existing IS events for each (pt, sl) × τ cell, then scores the same
RandomForest meta setup as ``compare_vertical_tau.py`` (uniqueness weights,
CPCV, purge/embargo = 1τ).

Mean CPCV logloss is the score we compute to see whether the meta
P(take-profit first) matched the labels (same as compare_vertical_tau.py).
[선정] AFML recommends logloss for scoring predicted probabilities.

Does **not** touch OOS. Requires precomputed daily bars + events under a run
root. Default multiplier: 1 (operating pt=sl=1σ). Pass ``--taus`` for a joint grid;
otherwise a single ``--tau`` (config default) is used.

Clock-agnostic: pass a treatment or control ``--run-root`` (do not mix).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.config import PipelineConfig
from src.cusum import select_events

from compare_vertical_tau import load_continuous_is, relabel_for_tau, score_tau


def _cell_key(pt: float, sl: float, tau: int) -> str:
    return f"{pt:g},{sl:g}@{tau}"


def main() -> None:
    config = PipelineConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[1.0],
        help="Symmetric pt=sl σ multiples (default: 1 = operating point)",
    )
    p.add_argument(
        "--tau",
        type=int,
        default=None,
        help="Single vertical barrier (default: config.vertical_bars). Ignored if --taus is set.",
    )
    p.add_argument(
        "--taus",
        type=int,
        nargs="+",
        default=None,
        help="Vertical barrier grid, crossed with --multipliers (e.g. 10 20 40 80)",
    )
    p.add_argument("--start", default=None, help="IS window start YYYY-MM-DD (inclusive)")
    p.add_argument("--end", default=None, help="IS window end YYYY-MM-DD (inclusive)")
    p.add_argument(
        "--event-mode",
        choices=("cusum", "every_bar"),
        default=config.event_mode,
        help="cusum = use saved events.csv; every_bar = rebuild from bars (default).",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    multipliers = [float(x) for x in args.multipliers]
    taus = [int(t) for t in (args.taus if args.taus else [args.tau or config.vertical_bars])]
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    print(
        f"pt/sl × τ CPCV | event_mode={args.event_mode} | multipliers={multipliers} | "
        f"taus={taus} | purge/embargo = 1τ bars (policy A) | IS only",
        flush=True,
    )
    bars, events, window = load_continuous_is(config=config, run_root=args.run_root, start=start, end=end)
    if "close_reason" in bars.columns:
        bars = bars.loc[bars["close_reason"] != "warmup"].reset_index(drop=True)
    if args.event_mode == "every_bar":
        cfg_ev = PipelineConfig(**{**config.__dict__, "event_mode": "every_bar"})
        events = select_events(bars, cfg_ev)
    print(
        f"loaded events={len(events)} bars={len(bars)} event_mode={args.event_mode} "
        f"days={window['n_days']} {window['first_day']}..{window['last_day']} "
        f"full_is={window['is_full_is']}",
        flush=True,
    )

    results: dict[str, dict] = {}
    n_cells = len(multipliers) * len(taus)
    i = 0
    for m in multipliers:
        for tau in taus:
            i += 1
            print(f"\n=== [{i}/{n_cells}] pt=sl={m:g} τ={tau} ===", flush=True)
            labeled = relabel_for_tau(events, bars, tau, config, pt=m, sl=m)
            cfg = PipelineConfig(**{**config.__dict__, "vertical_bars": tau, "pt": m, "sl": m})
            metrics = score_tau(labeled, cfg)
            results[_cell_key(m, m, tau)] = metrics
            print(json.dumps(metrics, indent=2), flush=True)

    ranked = [(k, cell) for k, cell in results.items() if "logloss_mean" in cell]
    ranked.sort(key=lambda x: x[1]["logloss_mean"])
    best = ranked[0][0] if ranked else None
    summary = {
        "window": f"{window['first_day']}..{window['last_day']}",
        "n_days": window["n_days"],
        "is_full_is": window["is_full_is"],
        "split": "is",
        "oos_touched": False,
        "selection_metric": "mean_cpcv_logloss",
        "boundary_policy": "A_purge_embargo_1tau",
        "multipliers": multipliers,
        "taus": taus,
        "event_mode": args.event_mode,
        "n_cpcv_groups": config.n_cpcv_groups,
        "n_cpcv_test_groups": config.n_cpcv_test_groups,
        "cpcv_paths": config.cpcv_path_count(),
        "n_bars": int(len(bars)),
        "n_events": int(len(events)),
        "results": results,
        "best_cell": best,
        "note": (
            "Selected by mean CPCV logloss on available IS events only; OOS untouched. "
            "Joint pt=sl × τ grid; each cell uses purge/embargo = 1τ. "
            f"event_mode={args.event_mode}. "
            "Partial IS is not a production lock."
            if not window["is_full_is"]
            else "Selected by mean CPCV logloss on IS only; OOS untouched. "
            "Joint pt=sl × τ grid; each cell uses purge/embargo = 1τ."
        ),
    }
    text = json.dumps(summary, indent=2)
    print("\n=== summary ===", flush=True)
    print(text, flush=True)
    out = args.out or (args.run_root / "pt_sl_cpcv.json")
    out.write_text(text)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
