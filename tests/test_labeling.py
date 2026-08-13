import numpy as np
import pandas as pd

from src.barriers import apply_triple_barrier
from src.cusum import cusum_events, every_bar_events, select_events
from src.pipeline import run_from_ticks
from tests.helpers import make_ticks, tight_config


def _monotonic_bars(n: int = 40, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    close = start + np.arange(n) * step
    ts = pd.date_range("2024-01-15", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "bar_id": np.arange(n),
            "start_ts": ts,
            "end_ts": ts + pd.Timedelta(seconds=50),
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "log_ret": np.concatenate([[np.nan], np.diff(np.log(close))]),
        }
    )


def test_cusum_detects_up_drift():
    bars = _monotonic_bars()
    events = cusum_events(bars, tight_config(cusum_mode="absolute", cusum_absolute_h=0.01))
    assert not events.empty
    assert set(events["cusum_side"].unique()) == {1}
    assert "side" not in events.columns


def test_cusum_resets_only_the_crossed_side():
    n = 8
    log_p = np.array([0.0, 0.02, 0.03, 0.01, -0.02, -0.05, -0.04, 0.0])
    close = np.exp(log_p)
    ts = pd.date_range("2024-01-15", periods=n, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {
            "bar_id": np.arange(n),
            "start_ts": ts,
            "end_ts": ts + pd.Timedelta(seconds=50),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "log_ret": np.concatenate([[np.nan], np.diff(log_p)]),
        }
    )
    events = cusum_events(bars, tight_config(cusum_mode="absolute", cusum_absolute_h=0.025))
    s_pos = s_neg = 0.0
    expected: list[tuple[int, int]] = []
    y = np.diff(log_p, prepend=log_p[0])
    for i in range(1, n):
        s_pos = max(0.0, s_pos + y[i])
        s_neg = min(0.0, s_neg + y[i])
        if s_neg < -0.025:
            expected.append((i, -1))
            s_neg = 0.0
        elif s_pos > 0.025:
            expected.append((i, 1))
            s_pos = 0.0
    assert expected
    assert list(zip(events["bar_id"].tolist(), events["cusum_side"].tolist())) == expected


def test_vol_scaled_cusum_lower_k_has_higher_recall():
    bars = _monotonic_bars(n=60, step=0.05)
    loose = cusum_events(bars, tight_config(cusum_mode="ewm_std", cusum_k=0.1, cusum_vol_span=10))
    strict = cusum_events(bars, tight_config(cusum_mode="ewm_std", cusum_k=1.0, cusum_vol_span=10))
    assert not loose.empty
    assert len(loose) >= len(strict)


def test_primary_side_is_flow_not_cusum_direction():
    bars = _monotonic_bars()
    bars["signed_flow"] = -1.0
    events = select_events(
        bars,
        tight_config(
            cusum_mode="absolute",
            cusum_absolute_h=0.01,
            primary_type="rule_bar_flow_sign",
        ),
    )
    assert not events.empty
    assert (events["cusum_side"] == 1).all()
    assert (events["side"] == -1).all()


def test_triple_barrier_take_profit_and_timeout():
    bars = _monotonic_bars(n=30, step=1.0)
    events = pd.DataFrame(
        {
            "bar_id": [0, 20],
            "event_ts": [bars.loc[0, "end_ts"], bars.loc[20, "end_ts"]],
            "side": [1, 1],
            "threshold": [0.01, 0.01],
        }
    )
    config = tight_config(pt=0.01, sl=0.01, vertical_bars=3, barrier_vol_span=5)
    labeled = apply_triple_barrier(bars, events, config)
    assert labeled.loc[0, "touch_type"] == "take_profit"
    assert int(labeled.loc[0, "y_meta"]) == 1
    assert labeled.loc[1, "touch_type"] in {"take_profit", "timeout"}


def test_pipeline_on_synthetic_ticks():
    ticks = make_ticks(n=500, buy_prob=0.9)
    bars, events, labeled, splits, _state = run_from_ticks(ticks, tight_config())
    assert len(bars) > 0
    assert set(labeled.columns) >= {"y_meta", "touch_type", "t1_ts", "side"}
    assert len(splits) >= 1
    train, test = splits[0]
    assert len(np.intersect1d(train, test)) == 0


def test_every_bar_events_follow_order_flow():
    ticks = make_ticks(n=200, buy_prob=1.0)
    bars, events, labeled, splits, _state = run_from_ticks(
        ticks, tight_config(event_mode="every_bar", bar_type="tick_imbalance")
    )
    usable = bars.loc[bars["close_reason"] != "warmup"]
    assert len(events) == len(usable)
    assert set(events["side"].unique()) == {1}


def test_research_session_labels_after_warmup():
    ticks = make_ticks(n=200, buy_prob=1.0)
    bars, events, labeled, splits, _state = run_from_ticks(
        ticks, tight_config(session="research", bar_type="tick_imbalance", event_mode="every_bar")
    )
    assert (bars["close_reason"] == "warmup").any()
    assert not labeled.empty
    assert len(labeled) == (bars["close_reason"] != "warmup").sum()


def test_explicit_warmup_session_skips_labels():
    ticks = make_ticks(n=120, buy_prob=1.0)
    bars, events, labeled, splits, state = run_from_ticks(
        ticks, tight_config(session="warmup", bar_type="tick_imbalance", initial_expected_ticks=40)
    )
    assert not bars.empty
    assert events.empty
    assert labeled.empty
    assert splits == []
    assert abs(state.b - 0.5) < 1e-9 or state.b > 0.5
