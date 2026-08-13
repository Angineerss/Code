"""Single source of truth for the labeling pipeline decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BarType = Literal["tick_imbalance", "volume_imbalance", "dollar_imbalance"]
CusumMode = Literal["ewm_std", "absolute"]
EventMode = Literal["cusum", "every_bar"]


@dataclass(frozen=True)
class PipelineConfig:
    # --- universe ---
    symbol: str = "BTCUSDT"
    market: str = "spot"
    data_type: str = "aggTrades"
    timestamp_storage: str = "UTC"
    session_filter: str | None = None  # crypto trades 24/7

    # --- imbalance bars ---
    bar_type: BarType = "dollar_imbalance"
    # E[θ]_0 = mean(prior 1y daily quote notional) / divisor. As-of day excluded.
    imbalance_divisor: int = 50
    imbalance_lookback_days: int = 365
    imbalance_ewma_span: int = 50
    initial_expected_ticks: int = 80  # fallback if prior-year trade counts are missing
    min_abs_2p1: float = 0.05
    max_abs_2p1: float = 0.15
    max_ticks_mult: float = 4.0
    expected_ticks_min_mult: float = 0.5
    expected_ticks_max_mult: float = 2.0
    # Keep E[θ] from drifting too far from the prior-year scale D.
    expected_imbalance_min_mult: float = 0.5
    expected_imbalance_max_mult: float = 2.0

    # --- event filter ---
    event_mode: EventMode = "cusum"
    cusum_mode: CusumMode = "ewm_std"
    cusum_vol_span: int = 50
    cusum_k: float = 0.1  # high-recall primary: h = 0.1 * EWM std of bar log-returns
    cusum_absolute_h: float = 0.001

    # --- triple barrier / meta label ---
    pt: float = 1.0
    sl: float = 1.0
    vertical_bars: int = 20
    barrier_vol_span: int = 50
    simultaneous_touch_y: int = 0
    timeout_y: int = 0

    # --- models / validation (ratios TBD after horizon is locked) ---
    primary_type: str = "rule_cusum_sign"
    primary_objective: str = "high_recall"
    meta_model: str = "random_forest"
    cv_method: str = "cpcv"
    n_cpcv_groups: int = 5
    n_cpcv_test_groups: int = 2
    purge_bars: int | None = None
    embargo_bars: int | None = None

    extra: dict = field(default_factory=dict)

    def resolved_purge_bars(self) -> int:
        return self.vertical_bars if self.purge_bars is None else self.purge_bars

    def resolved_embargo_bars(self) -> int:
        return self.vertical_bars if self.embargo_bars is None else self.embargo_bars
