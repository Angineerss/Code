"""Single source of truth for the labeling pipeline decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

BarType = Literal["tick_imbalance", "volume_imbalance", "dollar_imbalance"]
CusumMode = Literal["ewm_std", "absolute"]
EventMode = Literal["cusum", "every_bar"]
SessionType = Literal["warmup", "research"]
SplitName = Literal["is", "oos", "out_of_universe"]
PrimaryType = Literal["rule_bar_flow_sign", "rule_cusum_sign"]


@dataclass(frozen=True)
class PipelineConfig:
    # --- universe ---
    symbol: str = "BTCUSDT"
    market: str = "spot"
    data_type: str = "aggTrades"
    timestamp_storage: str = "UTC"
    session_filter: str | None = None  # crypto trades 24/7
    # Full tick universe (Vision aggTrades). OOS end is last published day at lock time.
    universe_start: date = date(2024, 1, 1)
    is_end: date = date(2025, 12, 31)
    oos_start: date = date(2026, 1, 1)
    oos_end: date = date(2026, 8, 13)

    # --- imbalance bars ---
    bar_type: BarType = "dollar_imbalance"
    # E[θ]_0 = mean(prior 1y daily quote notional) / divisor. Then EWMA-update.
    imbalance_divisor: int = 650
    imbalance_lookback_days: int = 365
    imbalance_ewma_span: int = 50
    initial_expected_ticks: int = 20_000  # init_T; ~50-100 bars/day on liquid BTC
    init_b: float = 0.5  # P[buy] seed; |2b-1| starts at 0
    session: SessionType = "research"
    min_abs_2p1: float = 0.05
    max_abs_2p1: float = 0.15
    max_ticks: int = 50_000  # hard cap on ticks per bar
    max_ticks_mult: float = 2.5  # 50,000 / 20,000; still clipped by max_ticks
    expected_ticks_min_mult: float = 0.5
    expected_ticks_max_mult: float = 2.0
    # Keep E[θ] from drifting too far from the prior-year scale D.
    expected_imbalance_min_mult: float = 0.5
    expected_imbalance_max_mult: float = 2.0

    # --- event filter (before the primary; not a trading signal) ---
    event_mode: EventMode = "cusum"
    cusum_mode: CusumMode = "ewm_std"
    cusum_vol_span: int = 50
    cusum_k: float = 1.0  # h = 1 * EWM std of bar log-returns (AFML vol-scaled threshold)
    cusum_absolute_h: float = 0.001

    # --- triple barrier / meta label ---
    pt: float = 1.0
    sl: float = 1.0
    vertical_bars: int = 20
    barrier_vol_span: int = 50
    simultaneous_touch_y: int = 0
    timeout_y: int = 0

    # --- models / validation (ratios TBD after horizon is locked) ---
    primary_type: PrimaryType = "rule_bar_flow_sign"
    primary_objective: str = "high_recall"
    meta_model: str = "random_forest"
    cv_method: str = "cpcv"
    n_cpcv_groups: int = 5
    n_cpcv_test_groups: int = 2
    purge_bars: int | None = None
    embargo_bars: int | None = None

    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.oos_start != self.is_end + timedelta(days=1):
            raise ValueError("OOS must start the UTC day after IS ends")
        if self.oos_start > self.oos_end:
            raise ValueError("OOS range is empty")
        if self.universe_start > self.is_end:
            raise ValueError("IS range is empty")

    def resolved_purge_bars(self) -> int:
        return self.vertical_bars if self.purge_bars is None else self.purge_bars

    def resolved_embargo_bars(self) -> int:
        return self.vertical_bars if self.embargo_bars is None else self.embargo_bars

    def split_for_day(self, day: date) -> SplitName:
        if day < self.universe_start or day > self.oos_end:
            return "out_of_universe"
        if day <= self.is_end:
            return "is"
        return "oos"

    def is_range(self) -> tuple[date, date]:
        return self.universe_start, self.is_end

    def oos_range(self) -> tuple[date, date]:
        return self.oos_start, self.oos_end
