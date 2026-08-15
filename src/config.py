"""Single source of truth for the labeling pipeline decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

BarType = Literal["tick_imbalance", "volume_imbalance", "dollar_imbalance"]
CusumMode = Literal["ewm_std", "absolute"]
EventMode = Literal["cusum", "every_bar"]
SessionType = Literal["warmup", "research"]
SplitName = Literal["warmup", "is", "oos", "out_of_universe"]
PrimaryType = Literal["rule_bar_flow_sign", "rule_cusum_sign"]


@dataclass(frozen=True)
class PipelineConfig:
    # --- universe ---
    symbol: str = "BTCUSDT"
    market: str = "spot"
    data_type: str = "aggTrades"
    timestamp_storage: str = "UTC"
    session_filter: str | None = None  # crypto trades 24/7
    # Full Vision aggTrades on disk (BTCUSDT listing → last locked published day).
    archive_start: date = date(2017, 8, 17)
    # First year after listing: build D lookback + EWMA only (no IS/OOS labels).
    warmup_end: date = date(2018, 8, 16)
    # Research IS starts the day after warmup (first day with a full 365d prior for D).
    universe_start: date = date(2018, 8, 17)
    is_end: date = date(2024, 12, 31)
    # Untouched holdout after IS. Learning/structuring must never use OOS
    # unless an explicit final-evaluation allow_oos flag is set.
    oos_start: date = date(2025, 1, 1)
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
    # Vertical barrier τ selected on IS via CPCV (results/vertical_tau_cpcv_*.json).
    # 2024H1 CPCV with purge/embargo=100 → τ=20 (min mean logloss).
    vertical_bars: int = 20
    barrier_vol_span: int = 50
    simultaneous_touch_y: int = 0
    timeout_y: int = 0

    # --- models / validation ---
    # Hypothesis (locked): betting is advantageous when taker dollar-flow imbalance
    # and a one-sided price run align (cusum_side == primary side).
    # CUSUM = timing filter; primary = flow direction; agreement = joint event gate.
    primary_type: PrimaryType = "rule_bar_flow_sign"
    require_cusum_flow_agree: bool = True
    primary_objective: str = "high_recall"
    # Meta features (locked): hypothesis strength + selected context.
    # Alignment itself is not a feature — it is enforced by require_cusum_flow_agree.
    meta_features: tuple[str, ...] = (
        "flow_strength",
        "cusum_excess_ratio",
        "tick_rel",
        "sigma",
    )
    meta_model: str = "random_forest"
    # All model/feature/threshold choices happen inside IS via CPCV only.
    # No extra IS holdout year and never tune on OOS.
    selection_method: str = "cpcv_only"
    cv_method: str = "cpcv"
    n_cpcv_groups: int = 6  # ~1y contiguous groups over ~6.4y IS
    n_cpcv_test_groups: int = 2  # C(6,2)=15 paths; train = remaining purged groups
    # CPCV / boundary purge-embargo lengths in imbalance bars.
    # Locked large enough that IS labels near OOS stay clear of the holdout.
    purge_bars: int | None = 100
    embargo_bars: int | None = 100
    # IS↔OOS boundary hygiene for meta-learning samples (AFML purge + embargo).
    boundary_purge: bool = True
    boundary_embargo: bool = True
    # Vertical barrier τ candidate grid for IS CPCV selection.
    vertical_tau_candidates: tuple[int, ...] = (10, 20, 40, 80)

    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.warmup_end < self.archive_start:
            raise ValueError("warmup_end must be on or after archive_start")
        if self.universe_start != self.warmup_end + timedelta(days=1):
            raise ValueError("IS must start the UTC day after warmup ends")
        if self.oos_start != self.is_end + timedelta(days=1):
            raise ValueError("OOS must start the UTC day after IS ends")
        if self.oos_start > self.oos_end:
            raise ValueError("OOS range is empty")
        if self.universe_start > self.is_end:
            raise ValueError("IS range is empty")
        if self.n_cpcv_groups < 2:
            raise ValueError("n_cpcv_groups must be >= 2")
        if not (1 <= self.n_cpcv_test_groups < self.n_cpcv_groups):
            raise ValueError("n_cpcv_test_groups must be in [1, n_cpcv_groups)")
        if self.selection_method != "cpcv_only":
            raise ValueError("selection_method must be 'cpcv_only'")
        if self.cv_method != "cpcv":
            raise ValueError("cv_method must be 'cpcv'")

    def resolved_purge_bars(self) -> int:
        return self.vertical_bars if self.purge_bars is None else self.purge_bars

    def resolved_embargo_bars(self) -> int:
        return self.vertical_bars if self.embargo_bars is None else self.embargo_bars

    def split_for_day(self, day: date) -> SplitName:
        if day < self.archive_start or day > self.oos_end:
            return "out_of_universe"
        if day <= self.warmup_end:
            return "warmup"
        if day <= self.is_end:
            return "is"
        return "oos"

    def warmup_range(self) -> tuple[date, date]:
        return self.archive_start, self.warmup_end

    def is_range(self) -> tuple[date, date]:
        return self.universe_start, self.is_end

    def oos_range(self) -> tuple[date, date]:
        return self.oos_start, self.oos_end

    def learning_range(self) -> tuple[date, date]:
        """Warmup + IS only. OOS is excluded from all learning/structuring defaults."""
        return self.archive_start, self.is_end

    def is_oos_day(self, day: date) -> bool:
        return self.split_for_day(day) == "oos"

    def assert_not_oos_day(self, day: date) -> None:
        """Raise if ``day`` is OOS. Learning pipelines must not touch OOS."""
        if self.is_oos_day(day):
            raise ValueError(
                f"OOS day {day.isoformat()} is locked untouched "
                f"(OOS={self.oos_start}..{self.oos_end}). "
                "Pass allow_oos only for the final evaluation run."
            )

    def assert_learning_range(self, start: date, end: date) -> None:
        """Raise if [start, end] intersects OOS. Default for structuring/training."""
        if end < start:
            raise ValueError("end must be on or after start")
        if start <= self.oos_end and end >= self.oos_start:
            raise ValueError(
                f"Range {start}..{end} intersects locked OOS "
                f"{self.oos_start}..{self.oos_end}. "
                "Learning/structuring must stay within "
                f"{self.archive_start}..{self.is_end} "
                "(use allow_oos only for final OOS evaluation)."
            )

    def cpcv_path_count(self) -> int:
        """Number of combinatorial CPCV paths: C(n_groups, n_test_groups)."""
        n = self.n_cpcv_groups
        k = self.n_cpcv_test_groups
        if k < 0 or k > n:
            return 0
        num = 1
        den = 1
        for i in range(k):
            num *= n - i
            den *= i + 1
        return num // den
