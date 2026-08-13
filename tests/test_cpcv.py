import numpy as np
import pandas as pd

from src.cpcv import cpcv_splits
from tests.helpers import tight_config


def test_cpcv_paths_are_purged():
    n = 40
    t0 = pd.date_range("2024-01-15", periods=n, freq="5min", tz="UTC")
    labeled = pd.DataFrame(
        {
            "event_ts": t0,
            "t1_ts": t0 + pd.Timedelta(minutes=10),
            "y_meta": np.resize([0, 1], n),
        }
    )
    config = tight_config(n_cpcv_groups=5, n_cpcv_test_groups=2, vertical_bars=2)
    paths = list(cpcv_splits(labeled, config))
    assert len(paths) == 10  # C(5,2)
    for train, test in paths:
        assert len(np.intersect1d(train, test)) == 0
        if train.size and test.size:
            # Purged train labels must not overlap the test time span.
            test_start = labeled.loc[test, "event_ts"].min()
            test_end = labeled.loc[test, "t1_ts"].max()
            train_t0 = labeled.loc[train, "event_ts"]
            train_t1 = labeled.loc[train, "t1_ts"]
            overlap = ~((train_t1 < test_start) | (train_t0 > test_end))
            assert not overlap.any()
