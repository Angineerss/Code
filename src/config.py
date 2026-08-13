"""Single source of truth for the labeling pipeline decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BarType = Literal["tick_imbalance", "volume_imbalance", "dollar_imbalance"]
CusumMode = Literal["ewm_std", "absolute"]


@dataclass(frozen=True)
class PipelineConfig:
    # --- universe ---
    symbol: str = "BTCUSDT"
    market: str = "spot"
    data_type: str = "aggTrades"
    # One complete UTC calendar day. Date is chosen at runtime (latest available).
    timestamp_storage: str = "UTC"
    session_filter: str | None = None  # crypto trades 24/7; no equity RTH filter

    # --- imbalance bars (AFML ch.2) ---
    bar_type: BarType = "dollar_imbalance"
    imbalance_ewma_span: int = 50
    initial_expected_ticks: int = 1_000

    # --- CUSUM event filter (high-recall primary) ---
    cusum_mode: CusumMode = "ewm_std"
    cusum_vol_span: int = 50
    cusum_k: float = 1.0  # h = k * EWM std of bar log-returns
    cusum_absolute_h: float = 0.001  # used only if cusum_mode == "absolute"

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
    purge_bars: int | None = None  # TBD; default in runner = vertical_bars
    embargo_bars: int | None = None  # TBD; default in runner = vertical_bars

    extra: dict = field(default_factory=dict)

    def resolved_purge_bars(self) -> int:
        return self.vertical_bars if self.purge_bars is None else self.purge_bars

    def resolved_embargo_bars(self) -> int:
        return self.vertical_bars if self.embargo_bars is None else self.embargo_bars
