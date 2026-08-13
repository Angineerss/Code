"""Run Binance tick → imbalance bars → CUSUM → triple-barrier labels."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .barriers import apply_triple_barrier
from .config import PipelineConfig
from .cpcv import cpcv_splits
from .cusum import select_events
from .download import load_or_download_day
from .imbalance import build_imbalance_bars


def run_from_ticks(ticks, config: PipelineConfig):
    bars = build_imbalance_bars(ticks, config)
    events = select_events(bars, config)
    labeled = apply_triple_barrier(bars, events, config)
    splits = list(cpcv_splits(labeled, config))
    return bars, events, labeled, splits


def _median(series) -> float | None:
    if series is None or len(series) == 0:
        return None
    value = series.median()
    return None if value != value else float(value)


def _summarize(bars, events, labeled, splits, config: PipelineConfig, day: date | None) -> dict:
    close_reasons = None if bars.empty or "close_reason" not in bars else bars["close_reason"].value_counts().to_dict()
    return {
        "symbol": config.symbol,
        "day": None if day is None else day.isoformat(),
        "bar_type": config.bar_type,
        "event_mode": config.event_mode,
        "cusum_k": config.cusum_k,
        "initial_expected_ticks": config.initial_expected_ticks,
        "n_bars": int(len(bars)),
        "n_events": int(len(events)),
        "n_labels": int(len(labeled)),
        "median_ticks_per_bar": _median(bars["tick_count"]) if not bars.empty else None,
        "median_bar_duration_s": _median(bars["duration_s"]) if not bars.empty and "duration_s" in bars else None,
        "close_reasons": close_reasons,
        "y_meta_rate": None if labeled.empty else float(labeled["y_meta"].mean()),
        "touch_types": None if labeled.empty else labeled["touch_type"].value_counts().to_dict(),
        "n_cpcv_paths": len(splits),
        "purge_bars": config.resolved_purge_bars(),
        "embargo_bars": config.resolved_embargo_bars(),
        "primary": config.primary_type,
        "meta_model": config.meta_model,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", default=None, help="UTC day YYYY-MM-DD; default = latest published")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--bar-type", default="dollar_imbalance")
    parser.add_argument("--event-mode", default="cusum", choices=("cusum", "every_bar"))
    args = parser.parse_args(argv)

    config = PipelineConfig(
        symbol=args.symbol.upper(),
        bar_type=args.bar_type,
        event_mode=args.event_mode,
        primary_type="rule_bar_flow_sign" if args.event_mode == "every_bar" else "rule_cusum_sign",
    )
    dest = Path(args.data_dir)
    day = date.fromisoformat(args.date) if args.date else None
    ticks, day = load_or_download_day(config.symbol, dest, config.market, day)
    bars, events, labeled, splits = run_from_ticks(ticks, config)

    dest.mkdir(parents=True, exist_ok=True)
    bars.to_csv(dest / f"{config.symbol}_{day}_imbalance_bars.csv", index=False)
    labeled.to_csv(dest / f"{config.symbol}_{day}_labels.csv", index=False)

    summary = _summarize(bars, events, labeled, splits, config, day)
    summary["n_ticks"] = int(len(ticks))
    print(json.dumps(summary, indent=2, default=str))
    (dest / f"{config.symbol}_{day}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
