import numpy as np
import pandas as pd

from src.features import META_FEATURE_NAMES, attach_meta_features, meta_feature_matrix
from src.pipeline import run_from_ticks
from tests.helpers import make_ticks, tight_config


def test_meta_feature_names_include_context():
    assert META_FEATURE_NAMES == (
        "flow_strength",
        "cusum_excess_ratio",
        "is_max_ticks",
        "duration_s",
        "tick_count",
        "sigma",
    )


def test_attach_meta_features_strength_and_context():
    bars = pd.DataFrame(
        {
            "bar_id": [0, 1],
            "signed_flow": [2_000_000.0, -1_500_000.0],
            "threshold": [1_000_000.0, 1_000_000.0],
            "close_reason": ["imbalance", "max_ticks"],
            "duration_s": [10.0, 20.0],
            "tick_count": [100, 500],
            "log_ret": [0.01, -0.02],
        }
    )
    events = pd.DataFrame(
        {
            "bar_id": [0, 1],
            "cusum_excess_ratio": [1.2, 1.5],
            "side": [1, -1],
        }
    )
    out = attach_meta_features(bars, events, vol_span=2)
    assert abs(out.loc[0, "flow_strength"] - 2.0) < 1e-9
    assert abs(out.loc[1, "flow_strength"] - 1.5) < 1e-9
    assert out.loc[0, "is_max_ticks"] == 0.0
    assert out.loc[1, "is_max_ticks"] == 1.0
    assert out.loc[0, "duration_s"] == 10.0
    assert out.loc[1, "tick_count"] == 500.0
    assert out["sigma"].notna().all()
    X = meta_feature_matrix(out)
    assert list(X.columns) == list(META_FEATURE_NAMES)


def test_pipeline_labels_include_locked_meta_features():
    ticks = make_ticks(n=500, buy_prob=0.9)
    _bars, events, labeled, _splits, _state = run_from_ticks(ticks, tight_config())
    assert not labeled.empty
    for name in META_FEATURE_NAMES:
        assert name in labeled.columns
        assert name in events.columns
    assert (labeled["flow_strength"] > 0).all()
    assert (labeled["cusum_excess_ratio"] >= 1.0).all()
    assert set(labeled["is_max_ticks"].unique()).issubset({0.0, 1.0})
    assert (labeled["duration_s"] > 0).all()
    assert (labeled["tick_count"] > 0).all()
    assert labeled["sigma"].notna().all()
