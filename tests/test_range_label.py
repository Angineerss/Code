from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from src.imbalance import ImbalanceSeed
from src.range_label import (
    concat_daily_bars,
    lookahead_end,
    purge_is_crossing_oos,
    run_range,
    utc_days,
)
from tests.helpers import make_ticks, tight_config


def _seed(_day: date) -> ImbalanceSeed:
    return ImbalanceSeed(expected_imbalance=5.0, expected_size=1.0)


def test_utc_days_inclusive():
    days = utc_days(date(2024, 1, 30), date(2024, 2, 1))
    assert days == [date(2024, 1, 30), date(2024, 1, 31), date(2024, 2, 1)]


def test_concat_recomputes_log_return_across_days():
    ts1 = pd.date_range("2024-01-15", periods=2, freq="1h", tz="UTC")
    ts2 = pd.date_range("2024-01-16", periods=2, freq="1h", tz="UTC")
    day1 = pd.DataFrame(
        {
            "close": [100.0, 101.0],
            "end_ts": ts1,
            "close_reason": ["imbalance", "imbalance"],
            "utc_day": ["2024-01-15", "2024-01-15"],
            "log_ret": [np.nan, np.log(101 / 100)],
            "bar_id": [0, 1],
        }
    )
    day2 = pd.DataFrame(
        {
            "close": [110.0, 111.0],
            "end_ts": ts2,
            "close_reason": ["imbalance", "imbalance"],
            "utc_day": ["2024-01-16", "2024-01-16"],
            "log_ret": [np.nan, np.log(111 / 110)],
            "bar_id": [0, 1],
        }
    )
    bars = concat_daily_bars([day1, day2])
    assert list(bars["bar_id"]) == [0, 1, 2, 3]
    assert np.isnan(bars.loc[0, "log_ret"])
    np.testing.assert_allclose(bars.loc[2, "log_ret"], np.log(110 / 101))


def test_purge_is_labels_whose_t1_crosses_oos():
    config = tight_config()
    labeled = pd.DataFrame(
        {
            "event_ts": pd.to_datetime(
                ["2025-12-31T20:00:00Z", "2025-12-30T10:00:00Z", "2026-01-01T01:00:00Z"]
            ),
            "t1_ts": pd.to_datetime(
                ["2026-01-01T02:00:00Z", "2025-12-30T12:00:00Z", "2026-01-01T03:00:00Z"]
            ),
            "y_meta": [1, 0, 1],
        }
    )
    kept = purge_is_crossing_oos(labeled, config)
    assert len(kept) == 2
    assert kept["event_ts"].dt.date.tolist() == [date(2025, 12, 30), date(2026, 1, 1)]


def test_lookahead_clips_to_oos_end():
    config = tight_config(barrier_lookahead_days=7)
    assert lookahead_end(config.is_end, config) == date(2026, 1, 7)
    assert lookahead_end(config.oos_end, config) == config.oos_end


def test_range_continues_ewma_and_skips_second_day_warmup():
    day1 = date(2024, 1, 15)
    day2 = date(2024, 1, 16)
    ticks = {
        day1: make_ticks(n=80, seed=1, start=datetime(2024, 1, 15, tzinfo=timezone.utc)),
        day2: make_ticks(n=80, seed=2, start=datetime(2024, 1, 16, tzinfo=timezone.utc)),
    }
    def load(day: date) -> pd.DataFrame:
        if day not in ticks:
            raise FileNotFoundError(day)
        return ticks[day]

    config = tight_config(event_mode="every_bar")
    bars, _events, labeled, is_labeled, splits, state, loaded = run_range(
        day1,
        day2,
        load,
        config,
        seed_for_day=_seed,
    )
    assert loaded == [day1, day2]
    assert (bars["utc_day"] == "2024-01-15").sum() > 0
    assert (bars["utc_day"] == "2024-01-16").sum() > 0
    assert (bars.loc[bars["utc_day"] == "2024-01-16", "close_reason"] != "warmup").all()
    assert (bars["close_reason"] == "warmup").sum() <= 1
    assert not labeled.empty
    assert set(labeled["split"].unique()) <= {"is"}
    assert len(is_labeled) == len(labeled)
    assert splits
    assert state is not None
    assert np.isfinite(state.expected_imbalance)


def test_range_barrier_can_close_on_next_day():
    ts = pd.date_range("2025-12-31 23:00", periods=4, freq="15min", tz="UTC")
    ts2 = pd.date_range("2026-01-01 00:00", periods=6, freq="15min", tz="UTC")
    close1 = np.array([100.0, 100.1, 100.2, 100.3])
    close2 = np.array([100.4, 102.0, 102.1, 102.2, 102.3, 102.4])
    day1 = pd.DataFrame(
        {
            "start_ts": ts,
            "end_ts": ts + pd.Timedelta(minutes=14),
            "open": close1,
            "high": close1,
            "low": close1,
            "close": close1,
            "close_reason": "imbalance",
            "signed_flow": 1.0,
            "threshold": 1.0,
            "utc_day": "2025-12-31",
        }
    )
    day2 = pd.DataFrame(
        {
            "start_ts": ts2,
            "end_ts": ts2 + pd.Timedelta(minutes=14),
            "open": close2,
            "high": close2,
            "low": close2,
            "close": close2,
            "close_reason": "imbalance",
            "signed_flow": 1.0,
            "threshold": 1.0,
            "utc_day": "2026-01-01",
        }
    )
    bars = concat_daily_bars([day1, day2])
    from src.range_label import label_from_bars, assign_split, clip_to_event_window

    config = tight_config(
        event_mode="every_bar",
        pt=0.01,
        sl=0.5,
        vertical_bars=8,
        barrier_vol_span=3,
    )
    _events, labeled = label_from_bars(bars, config)
    labeled = assign_split(labeled, config)
    labeled = clip_to_event_window(labeled, date(2025, 12, 31), date(2025, 12, 31))
    late = labeled.loc[labeled["event_ts"].dt.date == date(2025, 12, 31)]
    assert not late.empty
    assert (late["t1_ts"].dt.date == date(2026, 1, 1)).any()
    purged = purge_is_crossing_oos(late, config)
    assert len(purged) < len(late)
    assert (purged["t1_ts"].dt.date < date(2026, 1, 1)).all()
