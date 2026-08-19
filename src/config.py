"""Single source of truth for the labeling pipeline decisions.

Comment tags:
- ``[선정]`` — reason given in chat
- ``[임시값]`` — placeholder until a reason is written
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

BarType = Literal["dollar", "tick_imbalance", "volume_imbalance", "dollar_imbalance"]
CusumMode = Literal["ewm_std", "absolute"]
EventMode = Literal["cusum", "every_bar"]
SessionType = Literal["warmup", "research"]
SplitName = Literal["warmup", "is", "oos", "out_of_universe"]
PrimaryType = Literal["rule_bar_flow_sign", "rule_cusum_sign"]


@dataclass(frozen=True)
class PipelineConfig:
    # --- universe ---
    # [선정] BTCUSDT: long history of live ticks is easy to get.
    symbol: str = "BTCUSDT"
    # [임시값]
    market: str = "spot"
    # [선정] aggTrades: same-price prints are bundled; enough to recover aggressor side.
    data_type: str = "aggTrades"
    # [임시값]
    timestamp_storage: str = "UTC"
    # [임시값] crypto trades 24/7
    session_filter: str | None = None
    # [임시값] Vision listing → last locked published day.
    archive_start: date = date(2017, 8, 17)
    # [선정] one-year warmup calendar kept so IS:OOS dates (~8:2) do not move.
    # Listing day still bootstraps daily_T$ (no yesterday). Lookback=1.
    warmup_end: date = date(2018, 8, 16)
    # [선정] IS calendar; 8:2 vs OOS is the usual holdout mix.
    universe_start: date = date(2018, 8, 17)
    is_end: date = date(2024, 12, 31)
    # [선정] OOS unused in learning. Backtest after lock only (allow_oos).
    # Purge + embargo keep OOS out of learning samples. Train vs CV inside IS
    # is CPCV; hyperparameters and meta features are chosen on the CV paths.
    oos_start: date = date(2025, 1, 1)
    oos_end: date = date(2026, 8, 13)

    # --- information structure (clock) vs primary (direction) ---
    # [선정] Time bars do not represent trading activity (AFML). Dollar bars
    # close when cumulative quote hits T$, so sampling follows dollar volume.
    # Control (dollar_imbalance): close when |θ| ≥ E[θ] — contrast, not default.
    bar_type: BarType = "dollar"
    # [선정] Method B on IS months (100/200/400/650/1000). 400 vs 650 were
    # the less extreme pair; 650 is the faster of the two (~2.5 min bars vs
    # ~3.6). daily_T$ (spoken: 어제 조각) D = yesterday quote / 650.
    # T$ clip [0.5D, 2D] [선정]: keep bars increasing when today >> yesterday.
    # results/divisor_bar_screen.json
    imbalance_divisor: int = 650
    # [선정] Yesterday only. daily_T$ = 어제 조각; year-to-year notional jumps too much for 365d.
    imbalance_lookback_days: int = 1
    # [임시값]
    imbalance_ewma_span: int = 50
    # [선정] retune after bars so the first warmup bar is not far from typical tick_count.
    # [임시값] 20_000 until that pass.
    initial_expected_ticks: int = 20_000  # init_T
    # [선정] Control clock (dollar_imbalance) uses the AFML expected-imbalance
    # close: E[θ_T] ≈ E[T] × |2b-1| × E[size]. Tick-only imbalance is the
    # special case E[size]=1, i.e. E[θ_T] ≈ E[T] × |2b-1|. Treatment dollar
    # bars do not close or set strength from this.
    # [선정] init_b=0.5: b is P[buy]; start even so |2b-1| starts at 0.
    # Dollar path still EWMA-updates b for files / recorded threshold.
    init_b: float = 0.5
    # [임시값]
    session: SessionType = "research"
    # [선정] control clocks only. Clip |2b-1| into [0.05, 0.15]. At b=0.5,
    # |2b-1|=0 so E[θ]=0 and the bar would close on the first tick. Floor
    # keeps expected imbalance alive. Ceiling keeps E[θ] from growing so
    # large that |θ| never hits it.
    min_abs_2p1: float = 0.05
    max_abs_2p1: float = 0.15
    # [선정] force-close if a bar runs too many ticks; retune after bars exist.
    # [임시값] 50_000 until that pass. Dollar path uses this as-is.
    max_ticks: int = 50_000
    # [선정] control only: if |θ| never hits E[θ], force-close after a
    # multiple of E[T]: min(max_ticks, max(E[T]×mult, init_T)).
    # [임시값] 2.5 until that pass.
    max_ticks_mult: float = 2.5
    # [임시값] control E[T] clip; dollar does not EWMA E[T]
    expected_ticks_min_mult: float = 0.5
    expected_ticks_max_mult: float = 2.0
    # [선정] dollar T$: keep clip so a hot day vs yesterday makes more bars
    # (unclipped EWMA would grow T$ and bar count would stay flatter).
    # Long-run notional growth is already in yesterday's piece (lookback=1).
    # Same numbers clip control E[θ].
    expected_imbalance_min_mult: float = 0.5
    expected_imbalance_max_mult: float = 2.0

    # --- event filter (before the primary; not a trading signal) ---
    # [선정] every bar close is an event so the primary can keep recall high.
    event_mode: EventMode = "every_bar"
    # [임시값] contrast-only CUSUM
    cusum_mode: CusumMode = "ewm_std"
    cusum_vol_span: int = 50
    cusum_k: float = 1.0
    cusum_absolute_h: float = 0.001
    # [선정] AFML triple-barrier primary should maximize recall; do not drop weak |θ|.
    require_strong_imbalance: bool = False

    # --- triple barrier / meta label ---
    # [임시값]
    pt: float = 1.0
    sl: float = 1.0
    vertical_bars: int = 20
    # [임시값] one σ: same EWM series sizes pt/sl walls and is the meta
    # feature sigma (#2 · #23). One length; two uses.
    barrier_vol_span: int = 50
    # [선정] y_meta=1 only when a later bar hits TP and not SL on that same
    # bar. Intra-bar order is unknown, so same-bar both walls → 0.
    simultaneous_touch_y: int = 0
    timeout_y: int = 0
    # [선정] path uses bar high/low (not close-only) to see if that bar
    # touched a wall. Same-bar order is unknown; see y rule above.

    # --- models / validation ---
    # [선정] direction = sign(θ) = sign(signed_flow) on the dollar bar.
    primary_type: PrimaryType = "rule_bar_flow_sign"
    # [임시값]
    require_cusum_flow_agree: bool = False
    primary_objective: str = "high_recall"
    # [선정] strength = |θ| / that bar's quote — same dollar scale as the clock.
    # [임시값] sigma (barrier vol as context).
    meta_features: tuple[str, ...] = (
        "flow_strength",
        "sigma",
    )
    # [임시값] RF size not locked; 200 / depth 6 / leaf 10 is a placeholder.
    meta_model: str = "random_forest"
    # [선정] choices inside IS via CPCV only. OOS unused in learning (backtest after lock).
    selection_method: str = "cpcv_only"
    cv_method: str = "cpcv"
    # [선정] AFML recommends logloss for scoring predicted probabilities.
    # The score is a CPCV result, not a bar knob.
    selection_metric: str = "logloss"
    # [선정] AFML book example: 6 groups, 2 test → C(6,2)=15 paths.
    n_cpcv_groups: int = 6
    n_cpcv_test_groups: int = 2  # C(6,2)=15
    # [선정] use purge + embargo so OOS does not leak into learning.
    # [임시값] length = 1τ (follows vertical_bars).
    purge_bars: int | None = None
    embargo_bars: int | None = None
    # [선정] IS↔OOS boundary hygiene
    boundary_purge: bool = True
    boundary_embargo: bool = True
    # [임시값]
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
