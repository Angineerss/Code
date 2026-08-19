import numpy as np
import pandas as pd

from src.bar_clock_screen import (
    clip_stats,
    day_row,
    duration_stats,
    flow_strength,
    quiet_loud_stats,
    strength_stats,
    summarize_divisor,
    usable_bars,
)
from src.imbalance import EwmaState, ImbalanceSeed, build_dollar_bars
from tests.helpers import make_ticks, tight_config


def test_usable_bars_drop_warmup():
    bars = pd.DataFrame({"close_reason": ["warmup", "dollar", "dollar"]})
    assert list(usable_bars(bars)["close_reason"]) == ["dollar", "dollar"]


def test_flow_strength_is_abs_theta_over_quote():
    bars = pd.DataFrame(
        {
            "signed_flow": [10.0, -4.0],
            "quote_volume": [20.0, 8.0],
        }
    )
    got = flow_strength(bars)
    assert abs(got[0] - 0.5) < 1e-12
    assert abs(got[1] - 0.5) < 1e-12


def test_strength_stats_flags_pileup_at_edges():
    near0 = strength_stats(np.array([0.01, 0.02, 0.03]))
    near1 = strength_stats(np.array([0.96, 0.99, 1.0]))
    mid = strength_stats(np.array([0.2, 0.4, 0.6, 0.8]))
    assert near0["share_near_0"] == 1.0
    assert near1["share_near_1"] == 1.0
    assert mid["share_near_0"] == 0.0
    assert mid["share_near_1"] == 0.0


def test_duration_stats_flags_extreme_lengths():
    short = duration_stats(np.array([0.5, 1.0, 2.0]))
    long = duration_stats(np.array([5 * 3600, 6 * 3600]))
    assert short["share_shorter_than_5s"] == 1.0
    assert long["share_longer_than_4h"] == 1.0


def test_clip_stats_hits_upper_wall():
    d = 100.0
    got = clip_stats(np.array([200.0, 200.0, 150.0]), d)
    assert got["share_at_2d"] > 0.5
    assert got["median_t_over_d"] == 2.0


def test_quiet_loud_bar_counts_follow_quote():
    daily = pd.DataFrame(
        {
            "n_bars": [10] * 5 + [50] * 5,
            "daily_quote": [1e6] * 5 + [5e6] * 5,
        }
    )
    got = quiet_loud_stats(daily)
    assert got["quiet_median_bars"] == 10
    assert got["loud_median_bars"] == 50
    assert abs(got["loud_over_quiet_bars"] - 5.0) < 1e-9
    assert abs(got["loud_over_quiet_quote"] - 5.0) < 1e-9


def test_smaller_d_cuts_more_dollar_bars():
    ticks = make_ticks(n=800, buy_prob=0.75, qty=1.0, seed=5)
    cfg = tight_config(bar_type="dollar", max_ticks=10_000)
    warm = EwmaState(
        expected_ticks=20.0,
        b=0.7,
        expected_size=100.0,
        expected_imbalance=float("nan"),
        expected_dollar=float("nan"),
    )
    small = ImbalanceSeed(expected_imbalance=200.0, expected_size=100.0)
    large = ImbalanceSeed(expected_imbalance=2_000.0, expected_size=100.0)
    bars_small, _ = build_dollar_bars(ticks, cfg, seed=small, initial_state=warm)
    bars_large, _ = build_dollar_bars(ticks, cfg, seed=large, initial_state=warm)
    use_small = usable_bars(bars_small)
    use_large = usable_bars(bars_large)
    assert len(use_small) > len(use_large)
    row = day_row(
        day=__import__("datetime").date(2019, 1, 15),
        window="test",
        divisor=650,
        bars=use_small,
        daily_quote=float((ticks["price"] * ticks["qty"]).sum()),
        d_seed=200.0,
    )
    assert row["n_bars"] == len(use_small)
    assert row["_strength"].size == len(use_small)
    summary = summarize_divisor(
        pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} | {"divisor": 650}]),
        row["_duration"],
        row["_strength"],
    )
    assert summary["n_scored_days"] == 1
    assert summary["median_bars_per_day"] == float(len(use_small))
