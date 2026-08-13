import numpy as np
import pandas as pd

from src.barriers import apply_triple_barrier
from src.cusum import cusum_events, every_bar_events
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
    assert set(events["side"].unique()) == {1}


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
    bars, events, labeled, splits = run_from_ticks(ticks, tight_config())
    assert len(bars) > 0
    assert set(labeled.columns) >= {"y_meta", "touch_type", "t1_ts", "side"}
    assert len(splits) >= 1
    train, test = splits[0]
    assert len(np.intersect1d(train, test)) == 0


def test_every_bar_events_follow_order_flow():
    ticks = make_ticks(n=200, buy_prob=1.0)
    bars, events, labeled, splits = run_from_ticks(
        ticks, tight_config(event_mode="every_bar", bar_type="tick_imbalance")
    )
    assert len(events) == len(bars)
    assert set(events["side"].unique()) == {1}
    assert len(every_bar_events(bars)) == len(bars)
