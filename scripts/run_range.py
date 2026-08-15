"""Run dollar-imbalance bars → CUSUM over a UTC date range from the local archive."""

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

from src.config import PipelineConfig
from src.daily_notional import prior_year_notional
from src.download import load_day_from_archive
from src.imbalance import ImbalanceSeed
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
    args = parser.parse_args(argv)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be on or after --start")

    config = PipelineConfig(symbol=args.symbol.upper(), primary_type=args.primary)
    archive_dir = Path(args.archive_dir)
    out_dir = Path(args.out_dir)
    klines_dir = Path(args.klines_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    month_cache: dict[tuple[int, int], pd.DataFrame] = {}
    rows: list[dict] = []
    for day in daterange(start, end):
        print(f"[run] {day.isoformat()} ...", flush=True)
        ticks = load_day_from_archive(config.symbol, day, archive_dir, month_cache=month_cache)
        # Drop finished months from cache to bound memory.
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
        )
        seed = ImbalanceSeed(
            expected_imbalance=prior.threshold,
            expected_size=prior.expected_size,
        )
        bars, events, labeled, splits, state = run_from_ticks(
            ticks, config, seed=seed, initial_state=initial_state
        )
        bars.to_csv(out_dir / f"{config.symbol}_{day}_bars.csv", index=False)
        events.to_csv(out_dir / f"{config.symbol}_{day}_events.csv", index=False)
        labeled.to_csv(out_dir / f"{config.symbol}_{day}_labels.csv", index=False)
        (out_dir / f"{config.symbol}_{day}_ewma_state.json").write_text(
            json.dumps(asdict(state), indent=2)
        )
        summary = _summarize(bars, events, labeled, splits, config, day, ticks, prior, state)
        summary["n_ticks"] = int(len(ticks))
        summary["ewma_state_loaded_from"] = None if state_path is None else str(state_path)
        summary["ewma_continued"] = initial_state is not None
        summary["split"] = config.split_for_day(day)
        (out_dir / f"{config.symbol}_{day}_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        print(json.dumps(summary, indent=2, default=str), flush=True)
        rows.append(
            {
                "day": day.isoformat(),
                "split": summary["split"],
                "n_ticks": summary["n_ticks"],
                "n_bars": summary["n_bars"],
                "n_events": summary["n_events"],
                "close_reasons": summary["close_reasons"],
                "y_meta_rate": summary["y_meta_rate"],
                "ewma_continued": summary["ewma_continued"],
                "D": summary["imbalance_threshold_d"],
            }
        )

    manifest = {
        "symbol": config.symbol,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "archive_dir": str(archive_dir),
        "out_dir": str(out_dir),
        "days": rows,
    }
    (out_dir / f"{config.symbol}_{start}_{end}_range_summary.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(json.dumps(manifest, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
