"""Run Binance tick → imbalance bars → CUSUM → triple-barrier labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from .barriers import apply_triple_barrier
from .config import PipelineConfig
from .cpcv import cpcv_splits
from .cusum import select_events
from .daily_notional import PriorYearNotional, prior_year_notional
from .download import load_or_download_day
from .imbalance import EwmaState, ImbalanceSeed, build_bars, daily_quote_volume


def run_from_ticks(
    ticks,
    config: PipelineConfig,
    seed: ImbalanceSeed | None = None,
    initial_state: EwmaState | None = None,
):
    bars, state = build_bars(ticks, config, seed=seed, initial_state=initial_state)
    if config.session == "warmup":
        empty = bars.iloc[0:0]
        return bars, empty, empty, [], state
    usable = bars.loc[bars["close_reason"] != "warmup"].reset_index(drop=True)
    events = select_events(usable, config)
    labeled = apply_triple_barrier(usable, events, config)
    splits = list(cpcv_splits(labeled, config))
    return bars, events, labeled, splits, state


def load_ewma_state(path: Path) -> EwmaState:
    payload = json.loads(path.read_text())
    return EwmaState(
        expected_ticks=float(payload["expected_ticks"]),
        b=float(payload["b"]),
        expected_size=float(payload["expected_size"]),
        expected_imbalance=float(payload.get("expected_imbalance", float("nan"))),
    )


def resolve_ewma_state_path(dest: Path, symbol: str, day: date, explicit: str | None = None) -> Path | None:
    """Previous UTC day's saved EWMA, or an explicit path. Missing default is None."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"EWMA state not found: {path}")
        return path
    candidate = dest / f"{symbol}_{day - timedelta(days=1)}_ewma_state.json"
    return candidate if candidate.exists() else None


def _median(series) -> float | None:
    if series is None or len(series) == 0:
        return None
    value = series.median()
    return None if value != value else float(value)


def _primary_cusum_agree(labeled) -> float | None:
    if labeled is None or labeled.empty or "cusum_side" not in labeled.columns:
        return None
    both = labeled["cusum_side"].notna() & labeled["side"].notna()
    if not both.any():
        return None
    return float((labeled.loc[both, "side"] == labeled.loc[both, "cusum_side"]).mean())


def _summarize(
    bars,
    events,
    labeled,
    splits,
    config: PipelineConfig,
    day: date | None,
    ticks=None,
    prior: PriorYearNotional | None = None,
    state: EwmaState | None = None,
) -> dict:
    close_reasons = None if bars.empty or "close_reason" not in bars else bars["close_reason"].value_counts().to_dict()
    daily_notional = None if ticks is None else daily_quote_volume(ticks)
    return {
        "symbol": config.symbol,
        "day": None if day is None else day.isoformat(),
        "session": config.session,
        "bar_type": config.bar_type,
        "init_t": config.initial_expected_ticks,
        "init_b": config.init_b,
        "max_ticks": config.max_ticks,
        "imbalance_divisor": config.imbalance_divisor,
        "imbalance_lookback_days": config.imbalance_lookback_days,
        "prior_year_start": None if prior is None else prior.start.isoformat(),
        "prior_year_end": None if prior is None else prior.end.isoformat(),
        "prior_year_n_days": None if prior is None else prior.n_days,
        "prior_year_avg_daily_notional": None if prior is None else prior.average_daily_notional,
        "as_of_day_quote_notional": daily_notional,
        "imbalance_threshold_d": None if prior is None else prior.threshold,
        "ewma_state": None if state is None else asdict(state),
        "event_mode": config.event_mode,
        "cusum_k": config.cusum_k,
        "primary": config.primary_type,
        "n_bars": int(len(bars)),
        "n_events": int(len(events)),
        "n_labels": int(len(labeled)),
        "median_ticks_per_bar": _median(bars["tick_count"]) if not bars.empty else None,
        "median_bar_duration_s": _median(bars["duration_s"]) if not bars.empty and "duration_s" in bars else None,
        "close_reasons": close_reasons,
        "y_meta_rate": None if labeled.empty else float(labeled["y_meta"].mean()),
        "touch_types": None if labeled.empty else labeled["touch_type"].value_counts().to_dict(),
        "primary_cusum_agree_rate": _primary_cusum_agree(labeled),
        "n_cpcv_paths": len(splits),
        "purge_bars": config.resolved_purge_bars(),
        "embargo_bars": config.resolved_embargo_bars(),
        "meta_model": config.meta_model,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", default=None, help="UTC day YYYY-MM-DD; default = latest published")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--archive-dir",
        default=None,
        help="Local Vision archive root with monthly/ and daily/; if set, load ticks from there",
    )
    parser.add_argument(
        "--bar-type",
        default="dollar_imbalance",
        choices=("tick_imbalance", "volume_imbalance", "dollar_imbalance"),
    )
    parser.add_argument("--event-mode", default="cusum", choices=("cusum", "every_bar"))
    parser.add_argument(
        "--primary",
        default="rule_bar_flow_sign",
        choices=("rule_bar_flow_sign", "rule_cusum_sign"),
        help="Primary side after the event filter. Default = bar signed-flow sign, not CUSUM direction.",
    )
    parser.add_argument("--session", default="research", choices=("warmup", "research"))
    parser.add_argument(
        "--ewma-state",
        default=None,
        help="Previous day's ewma_state.json; default = data/{symbol}_{date-1}_ewma_state.json",
    )
    args = parser.parse_args(argv)

    config = PipelineConfig(
        symbol=args.symbol.upper(),
        bar_type=args.bar_type,
        event_mode=args.event_mode,
        session=args.session,
        primary_type=args.primary,
    )
    dest = Path(args.data_dir)
    day = date.fromisoformat(args.date) if args.date else None
    archive_dir = Path(args.archive_dir) if args.archive_dir else None
    ticks, day = load_or_download_day(
        config.symbol, dest, config.market, day, archive_dir=archive_dir
    )

    state_path = resolve_ewma_state_path(dest, config.symbol, day, args.ewma_state)
    initial_state = None if state_path is None else load_ewma_state(state_path)

    prior = None
    seed = None
    if config.bar_type == "dollar_imbalance":
        prior = prior_year_notional(
            config.symbol,
            day,
            divisor=config.imbalance_divisor,
            lookback_days=config.imbalance_lookback_days,
            cache_dir=dest / "klines",
            listing_date=config.archive_start,
        )
        seed = ImbalanceSeed(
            expected_imbalance=prior.threshold,
            expected_size=prior.expected_size,
        )
    bars, events, labeled, splits, state = run_from_ticks(
        ticks, config, seed=seed, initial_state=initial_state
    )

    dest.mkdir(parents=True, exist_ok=True)
    bars.to_csv(dest / f"{config.symbol}_{day}_bars.csv", index=False)
    (dest / f"{config.symbol}_{day}_ewma_state.json").write_text(
        json.dumps(asdict(state), indent=2)
    )
    if config.session != "warmup":
        labeled.to_csv(dest / f"{config.symbol}_{day}_labels.csv", index=False)

    summary = _summarize(bars, events, labeled, splits, config, day, ticks, prior, state)
    summary["n_ticks"] = int(len(ticks))
    summary["ewma_state_loaded_from"] = None if state_path is None else str(state_path)
    summary["ewma_continued"] = initial_state is not None
    print(json.dumps(summary, indent=2, default=str))
    (dest / f"{config.symbol}_{day}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
