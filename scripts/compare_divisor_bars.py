#!/usr/bin/env python3
"""Method B: pick a dollar-bar divisor from bar shape, not from CPCV.

Builds dollar bars for several divisors on the same ticks (one load per UTC
day). Compares bar duration, flow_strength, and how bar counts move on quiet
vs busy days. Labels, CPCV, and OOS are not touched.

Lock (chat): ``imbalance_divisor=650``. Candidates 100/200/400/650/1000 on
154 IS days; 400 vs 650 were the less extreme pair; 650 is the faster of
those two. T$ clip ``[0.5D, 2D]`` kept: more bars when today >> yesterday.
Output: ``results/divisor_bar_screen.json``.

Default windows are a few IS months (quiet / crash / busy / mid / ETF). Each
window prepends ``--burn-in-days`` of EWMA-only days that are not scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bar_clock_screen import day_row, summarize_divisor
from src.config import PipelineConfig
from src.daily_notional import prior_year_notional
from src.download import load_day_from_archive
from src.imbalance import ImbalanceSeed, build_dollar_bars, daily_quote_volume

# IS months only. Quiet / crash / busy / mid / ETF-era activity.
DEFAULT_WINDOWS = (
    (date(2019, 1, 1), date(2019, 1, 31)),
    (date(2020, 3, 1), date(2020, 3, 31)),
    (date(2021, 5, 1), date(2021, 5, 31)),
    (date(2023, 6, 1), date(2023, 6, 30)),
    (date(2024, 3, 1), date(2024, 3, 31)),
)
DEFAULT_DIVISORS = (100, 200, 400, 650, 1000)


def daterange(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def parse_divisors(text: str) -> tuple[int, ...]:
    vals = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("divisors must be positive integers")
    return vals


def parse_windows(text: str | None) -> list[tuple[date, date]]:
    if not text:
        return list(DEFAULT_WINDOWS)
    out: list[tuple[date, date]] = []
    for part in text.split(","):
        a, b = part.split(":")
        start, end = date.fromisoformat(a.strip()), date.fromisoformat(b.strip())
        if end < start:
            raise ValueError(f"window {part} has end before start")
        out.append((start, end))
    return out


def window_label(start: date, end: date) -> str:
    return f"{start.isoformat()}_{end.isoformat()}"


def main(argv: list[str] | None = None) -> int:
    base = PipelineConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--divisors", default=",".join(str(d) for d in DEFAULT_DIVISORS))
    p.add_argument(
        "--windows",
        default=None,
        help="Comma-separated UTC windows start:end. Default: quiet/crash/busy/mid/ETF months.",
    )
    p.add_argument("--start", default=None, help="Single-window start (with --end).")
    p.add_argument("--end", default=None, help="Single-window end (with --start).")
    p.add_argument("--burn-in-days", type=int, default=7)
    p.add_argument("--symbol", default=base.symbol)
    p.add_argument("--archive-dir", default="data/aggtrades")
    p.add_argument("--klines-dir", default="data/klines")
    p.add_argument("--out", type=Path, default=Path("results/divisor_bar_screen.json"))
    args = p.parse_args(argv)

    if (args.start is None) ^ (args.end is None):
        raise SystemExit("pass both --start and --end, or neither")
    windows = (
        [(date.fromisoformat(args.start), date.fromisoformat(args.end))]
        if args.start
        else parse_windows(args.windows)
    )
    divisors = parse_divisors(args.divisors)
    burn = max(int(args.burn_in_days), 0)
    symbol = args.symbol.upper()
    archive_dir = Path(args.archive_dir)
    klines_dir = Path(args.klines_dir)

    for start, end in windows:
        burn_start = start - timedelta(days=burn)
        base.assert_learning_range(burn_start, end)
        if burn_start < base.archive_start:
            raise SystemExit(
                f"burn-in for {start} starts {burn_start}, before archive {base.archive_start}"
            )

    month_cache: dict[tuple[int, int], dict[date, pd.DataFrame]] = {}
    daily_rows: list[dict] = []
    pooled_duration: dict[int, list[np.ndarray]] = {d: [] for d in divisors}
    pooled_strength: dict[int, list[np.ndarray]] = {d: [] for d in divisors}

    for start, end in windows:
        label = window_label(start, end)
        burn_start = start - timedelta(days=burn)
        states: dict[int, object] = {d: None for d in divisors}
        print(f"[window] {label} burn-in {burn_start.isoformat()}..{start.isoformat()}", flush=True)
        for day in daterange(burn_start, end):
            scoring = day >= start
            print(f"[{'score' if scoring else 'burn'}] {day.isoformat()}", flush=True)
            ticks = load_day_from_archive(symbol, day, archive_dir, month_cache=month_cache)
            keep = (day.year, day.month)
            for key in list(month_cache):
                if key != keep:
                    del month_cache[key]
            quote = float(daily_quote_volume(ticks))
            for divisor in divisors:
                cfg = replace(base, symbol=symbol, imbalance_divisor=int(divisor), bar_type="dollar")
                prior = prior_year_notional(
                    symbol,
                    day,
                    divisor=int(divisor),
                    lookback_days=cfg.imbalance_lookback_days,
                    cache_dir=klines_dir,
                    listing_date=cfg.archive_start,
                )
                seed = ImbalanceSeed(
                    expected_imbalance=prior.threshold,
                    expected_size=prior.expected_size,
                )
                bars, state = build_dollar_bars(
                    ticks, cfg, seed=seed, initial_state=states[divisor]
                )
                states[divisor] = state
                if not scoring:
                    continue
                row = day_row(day, label, int(divisor), bars, quote, float(prior.threshold))
                pooled_duration[divisor].append(row.pop("_duration"))
                pooled_strength[divisor].append(row.pop("_strength"))
                row.pop("_threshold", None)
                daily_rows.append(row)

    daily = pd.DataFrame(daily_rows)
    summaries = []
    for divisor in divisors:
        part = daily.loc[daily["divisor"] == divisor].copy() if not daily.empty else daily
        dur = (
            np.concatenate(pooled_duration[divisor])
            if pooled_duration[divisor]
            else np.array([], dtype=float)
        )
        st = (
            np.concatenate(pooled_strength[divisor])
            if pooled_strength[divisor]
            else np.array([], dtype=float)
        )
        summaries.append(summarize_divisor(part, dur, st))

    payload = {
        "method": "B_bar_shape",
        "bar_type": "dollar",
        "lookback_days": base.imbalance_lookback_days,
        "divisors": list(divisors),
        "windows": [[a.isoformat(), b.isoformat()] for a, b in windows],
        "burn_in_days": burn,
        "n_scored_days": 0 if daily.empty else int(daily["day"].nunique()),
        "oos_touched": False,
        "note": (
            "Screening used to lock imbalance_divisor=650 (faster of 400 vs 650). "
            "T$ clip [0.5D, 2D] kept so a hot day vs yesterday makes more bars. "
            "Same ticks, one pass per day, separate EWMA per divisor per window. "
            "No labels / CPCV / OOS."
        ),
        "by_divisor": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    if not daily.empty:
        daily.to_csv(csv_path, index=False)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    if not daily.empty:
        print(f"wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
