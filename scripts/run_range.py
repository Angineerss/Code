"""Run dollar-imbalance bars (optionally → CUSUM/labels) over a UTC date range."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.checkpoint import (
    day_is_complete,
    push_checkpoints,
    restore_checkpoints,
    save_day_checkpoint,
)
from src.config import PipelineConfig
from src.daily_notional import prior_year_notional
from src.download import load_day_from_archive
from src.imbalance import ImbalanceSeed, build_bars
from src.pipeline import load_ewma_state, resolve_ewma_state_path, run_from_ticks, _summarize


def daterange(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def main(argv: list[str] | None = None) -> int:
    config = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=config.symbol)
    parser.add_argument("--start", required=True, help="UTC start YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="UTC end YYYY-MM-DD")
    parser.add_argument("--archive-dir", default="data/aggtrades")
    parser.add_argument("--out-dir", default="data/runs")
    parser.add_argument("--klines-dir", default="data/klines")
    parser.add_argument(
        "--primary",
        default=config.primary_type,
        choices=("rule_bar_flow_sign", "rule_cusum_sign"),
    )
    parser.add_argument(
        "--bars-only",
        action="store_true",
        help="Build dollar imbalance bars + EWMA only (skip CUSUM/labels)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip days that already have EWMA (and labels, unless --bars-only)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="results/checkpoints",
        help="Tracked dir for EWMA/labels/progress (survives data/ wipes)",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Do not copy day artifacts into --checkpoint-dir",
    )
    parser.add_argument(
        "--push-checkpoint",
        action="store_true",
        help="git commit + push results/checkpoints after --checkpoint-every days",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Push a checkpoint every N newly finished days (default 1)",
    )
    parser.add_argument(
        "--allow-oos",
        action="store_true",
        help="Permit OOS dates. Default forbids any OOS day (learning/structuring).",
    )
    args = parser.parse_args(argv)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be on or after --start")

    config = PipelineConfig(symbol=args.symbol.upper(), primary_type=args.primary)
    if not args.allow_oos:
        try:
            config.assert_learning_range(start, end)
        except ValueError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            return 2
    archive_dir = Path(args.archive_dir)
    out_dir = Path(args.out_dir)
    klines_dir = Path(args.klines_dir)
    checkpoint_root = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_checkpoint:
        n_restored = restore_checkpoints(checkpoint_root, out_dir)
        if n_restored:
            print(f"[checkpoint] restored {n_restored} files → {out_dir}", flush=True)

    month_cache: dict[tuple[int, int], pd.DataFrame] = {}
    rows: list[dict] = []
    finished_since_push = 0
    for day in daterange(start, end):
        if args.skip_existing and day_is_complete(
            out_dir, config.symbol, day, bars_only=bool(args.bars_only)
        ):
            print(f"[skip] {day.isoformat()}", flush=True)
            try:
                summary = json.loads((out_dir / f"{config.symbol}_{day}_summary.json").read_text())
            except FileNotFoundError:
                summary = {
                    "day": day.isoformat(),
                    "split": config.split_for_day(day),
                    "skipped": True,
                }
            rows.append(
                {
                    "day": day.isoformat(),
                    "split": summary.get("split", config.split_for_day(day)),
                    "n_ticks": summary.get("n_ticks"),
                    "n_bars": summary.get("n_bars"),
                    "n_events": summary.get("n_events"),
                    "close_reasons": summary.get("close_reasons"),
                    "y_meta_rate": summary.get("y_meta_rate"),
                    "ewma_continued": summary.get("ewma_continued"),
                    "D": summary.get("imbalance_threshold_d"),
                    "skipped": True,
                }
            )
            continue

        print(f"[run] {day.isoformat()} ...", flush=True)
        if not args.allow_oos:
            try:
                config.assert_not_oos_day(day)
            except ValueError as exc:
                print(str(exc), file=sys.stderr, flush=True)
                return 2
        ticks = load_day_from_archive(config.symbol, day, archive_dir, month_cache=month_cache)
        keep = (day.year, day.month)
        for key in list(month_cache):
            if key != keep:
                del month_cache[key]

        state_path = resolve_ewma_state_path(out_dir, config.symbol, day)
        initial_state = None if state_path is None else load_ewma_state(state_path)
        prior = prior_year_notional(
            config.symbol,
            day,
            divisor=config.imbalance_divisor,
            lookback_days=config.imbalance_lookback_days,
            cache_dir=klines_dir,
            listing_date=config.archive_start,
        )
        seed = ImbalanceSeed(
            expected_imbalance=prior.threshold,
            expected_size=prior.expected_size,
        )

        if args.bars_only:
            bars, state = build_bars(ticks, config, seed=seed, initial_state=initial_state)
            empty = bars.iloc[0:0]
            events, labeled, splits = empty, empty, []
        else:
            bars, events, labeled, splits, state = run_from_ticks(
                ticks, config, seed=seed, initial_state=initial_state
            )

        bars_path = out_dir / f"{config.symbol}_{day}_bars.csv"
        ewma_path = out_dir / f"{config.symbol}_{day}_ewma_state.json"
        bars.to_csv(bars_path, index=False)
        if not args.bars_only:
            events.to_csv(out_dir / f"{config.symbol}_{day}_events.csv", index=False)
            labeled.to_csv(out_dir / f"{config.symbol}_{day}_labels.csv", index=False)
        ewma_path.write_text(json.dumps(asdict(state), indent=2))
        summary = _summarize(bars, events, labeled, splits, config, day, ticks, prior, state)
        summary["n_ticks"] = int(len(ticks))
        summary["ewma_state_loaded_from"] = None if state_path is None else str(state_path)
        summary["ewma_continued"] = initial_state is not None
        summary["split"] = config.split_for_day(day)
        summary["bars_only"] = bool(args.bars_only)
        (out_dir / f"{config.symbol}_{day}_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        print(
            json.dumps(
                {
                    "day": summary["day"],
                    "split": summary["split"],
                    "n_ticks": summary["n_ticks"],
                    "n_bars": summary["n_bars"],
                    "close_reasons": summary["close_reasons"],
                    "ewma_continued": summary["ewma_continued"],
                    "D": summary["imbalance_threshold_d"],
                    "bars_only": summary["bars_only"],
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        rows.append(
            {
                "day": day.isoformat(),
                "split": summary["split"],
                "n_ticks": summary["n_ticks"],
                "n_bars": summary["n_bars"],
                "n_events": summary.get("n_events"),
                "close_reasons": summary["close_reasons"],
                "y_meta_rate": summary.get("y_meta_rate"),
                "ewma_continued": summary["ewma_continued"],
                "D": summary["imbalance_threshold_d"],
                "skipped": False,
            }
        )
        # Progress manifest so a long run is inspectable mid-flight.
        progress = {
            "symbol": config.symbol,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "bars_only": bool(args.bars_only),
            "last_day": day.isoformat(),
            "n_days_done": len(rows),
            "days": rows,
        }
        (out_dir / f"{config.symbol}_{start}_{end}_progress.json").write_text(
            json.dumps(progress, indent=2, default=str)
        )
        if not args.no_checkpoint:
            save_day_checkpoint(
                out_dir,
                checkpoint_root,
                config.symbol,
                day,
                start=start,
                end=end,
            )
            finished_since_push += 1
            every = max(int(args.checkpoint_every), 1)
            if args.push_checkpoint and finished_since_push >= every:
                status = push_checkpoints(
                    ROOT,
                    checkpoint_root,
                    f"checkpoint: {out_dir.name} last_day={day.isoformat()}",
                )
                print(f"[checkpoint] {status}", flush=True)
                finished_since_push = 0

    manifest = {
        "symbol": config.symbol,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "archive_dir": str(archive_dir),
        "out_dir": str(out_dir),
        "bars_only": bool(args.bars_only),
        "days": rows,
    }
    (out_dir / f"{config.symbol}_{start}_{end}_range_summary.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    if not args.no_checkpoint:
        save_day_checkpoint(
            out_dir,
            checkpoint_root,
            config.symbol,
            end,
            start=start,
            end=end,
        )
        if args.push_checkpoint and finished_since_push:
            status = push_checkpoints(
                ROOT,
                checkpoint_root,
                f"checkpoint: {out_dir.name} range_done {start}..{end}",
            )
            print(f"[checkpoint] {status}", flush=True)
    print(json.dumps({"done": True, "n_days": len(rows)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
