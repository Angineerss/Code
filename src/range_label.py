"""Label a UTC date range on one continuous bar clock.

Daily EWMA still updates per tick day. CUSUM and triple-barrier then run on the
concatenated bars so S± and τ can cross midnight. IS labels whose t1 lands in
OOS are dropped (boundary purge). CPCV is computed on remaining IS labels only.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .barriers import apply_triple_barrier
from .config import PipelineConfig
from .cpcv import cpcv_splits
from .cusum import select_events
from .imbalance import EwmaState, ImbalanceSeed, build_bars

TickLoader = Callable[[date], pd.DataFrame]
SeedLoader = Callable[[date], ImbalanceSeed | None]


def utc_days(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end must be on or after start")
    days: list[date] = []
    day = start
    while day <= end:
        days.append(day)
        day += timedelta(days=1)
    return days


def utc_date(ts) -> date:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC")
    return stamp.date()


def concat_daily_bars(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack daily bars and recompute log-returns / bar_id on the joined clock."""
    frames = [part for part in parts if part is not None and not part.empty]
    if not frames:
        return pd.DataFrame()
    bars = pd.concat(frames, ignore_index=True)
    close = bars["close"].to_numpy(dtype=float)
    bars["log_ret"] = np.concatenate([[np.nan], np.diff(np.log(close))])
    bars["bar_id"] = np.arange(len(bars), dtype=np.int64)
    return bars


def label_from_bars(bars: pd.DataFrame, config: PipelineConfig):
    if bars.empty:
        empty = bars.iloc[0:0]
        return empty, empty
    usable = bars.loc[bars["close_reason"] != "warmup"].copy().reset_index(drop=True)
    if usable.empty:
        return usable, usable
    usable["bar_id"] = np.arange(len(usable), dtype=np.int64)
    close = usable["close"].to_numpy(dtype=float)
    usable["log_ret"] = np.concatenate([[np.nan], np.diff(np.log(close))])
    events = select_events(usable, config)
    labeled = apply_triple_barrier(usable, events, config)
    return events, labeled


def assign_split(labeled: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    out = labeled.copy()
    if out.empty:
        out["split"] = pd.Series(dtype="object")
        return out
    days = out["event_ts"].map(utc_date)
    out["split"] = [config.split_for_day(day) for day in days]
    return out


def purge_is_crossing_oos(labeled: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Drop IS events whose triple-barrier t1 is on or after OOS start."""
    if labeled.empty:
        return labeled.reset_index(drop=True)
    event_day = labeled["event_ts"].map(utc_date)
    t1_day = labeled["t1_ts"].map(utc_date)
    is_event = [(config.universe_start <= day <= config.is_end) for day in event_day]
    crosses = [day >= config.oos_start for day in t1_day]
    keep = [not (inside and cross) for inside, cross in zip(is_event, crosses)]
    return labeled.loc[keep].reset_index(drop=True)


def clip_to_event_window(
    labeled: pd.DataFrame,
    event_start: date,
    event_end: date,
) -> pd.DataFrame:
    if labeled.empty:
        return labeled.reset_index(drop=True)
    days = labeled["event_ts"].map(utc_date)
    keep = [(event_start <= day <= event_end) for day in days]
    return labeled.loc[keep].reset_index(drop=True)


def lookahead_end(event_end: date, config: PipelineConfig) -> date:
    extra = event_end + timedelta(days=config.barrier_lookahead_days)
    return extra if extra <= config.oos_end else config.oos_end


def build_range_bars(
    days: list[date],
    load_ticks: TickLoader,
    config: PipelineConfig,
    seed_for_day: SeedLoader | None = None,
    initial_state: EwmaState | None = None,
    dest: Path | None = None,
    required: bool = True,
) -> tuple[pd.DataFrame, EwmaState | None, list[date]]:
    """Build imbalance bars day by day, continuing EWMA. ``required`` days must load."""
    state = initial_state
    parts: list[pd.DataFrame] = []
    loaded: list[date] = []
    for day in days:
        try:
            ticks = load_ticks(day)
        except FileNotFoundError:
            if required:
                raise
            continue
        seed = seed_for_day(day) if seed_for_day is not None else None
        bars, state = build_bars(ticks, config, seed=seed, initial_state=state)
        if dest is not None and state is not None:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / f"{config.symbol}_{day}_ewma_state.json").write_text(
                json.dumps(asdict(state), indent=2)
            )
        if not bars.empty:
            bars = bars.copy()
            bars["utc_day"] = day.isoformat()
            parts.append(bars)
        loaded.append(day)
    return concat_daily_bars(parts), state, loaded


def run_range(
    event_start: date,
    event_end: date,
    load_ticks: TickLoader,
    config: PipelineConfig,
    seed_for_day: SeedLoader | None = None,
    initial_state: EwmaState | None = None,
    dest: Path | None = None,
):
    """Bars for [event_start, lookahead_end], labels kept for the event window."""
    event_days = utc_days(event_start, event_end)
    bar_end = lookahead_end(event_end, config)
    extra_days = utc_days(event_end + timedelta(days=1), bar_end) if bar_end > event_end else []
    event_bars, state, loaded_events = build_range_bars(
        event_days,
        load_ticks,
        config,
        seed_for_day=seed_for_day,
        initial_state=initial_state,
        dest=dest,
        required=True,
    )
    extra_bars, state, loaded_extra = build_range_bars(
        extra_days,
        load_ticks,
        config,
        seed_for_day=seed_for_day,
        initial_state=state,
        dest=dest,
        required=False,
    )
    frames = [part for part in (event_bars, extra_bars) if part is not None and not part.empty]
    bars = concat_daily_bars(frames) if len(frames) > 1 else (frames[0] if frames else event_bars)
    if config.session == "warmup":
        empty = bars.iloc[0:0] if not bars.empty else bars
        return bars, empty, empty, empty, [], state, loaded_events + loaded_extra
    events, labeled = label_from_bars(bars, config)
    labeled = clip_to_event_window(labeled, event_start, event_end)
    labeled = assign_split(labeled, config)
    labeled = purge_is_crossing_oos(labeled, config)
    is_labeled = labeled.loc[labeled["split"] == "is"].reset_index(drop=True) if not labeled.empty else labeled
    splits = list(cpcv_splits(is_labeled, config))
    return bars, events, labeled, is_labeled, splits, state, loaded_events + loaded_extra
