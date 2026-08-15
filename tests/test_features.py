import pandas as pd

from src.features import META_FEATURE_NAMES, attach_meta_features, meta_feature_matrix
from src.pipeline import run_from_ticks
from tests.helpers import make_ticks, tight_config


def test_meta_feature_names_locked_set():
    assert META_FEATURE_NAMES == (
        "flow_strength",
        "cusum_excess_ratio",
        "tick_rel",
        "sigma",
    )
    assert "tick_count" not in META_FEATURE_NAMES


def test_attach_meta_features_tick_rel_over_expected_ticks():
    bars = pd.DataFrame(
        {
            "bar_id": [0, 1],
            "signed_flow": [2_000_000.0, -1_500_000.0],
            "threshold": [1_000_000.0, 1_000_000.0],
            "close_reason": ["imbalance", "max_ticks"],
            "tick_count": [100, 500],
            "expected_ticks": [200.0, 250.0],
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
    assert abs(out.loc[0, "tick_rel"] - 0.5) < 1e-9
    assert abs(out.loc[1, "tick_rel"] - 2.0) < 1e-9
    assert list(meta_feature_matrix(out).columns) == list(META_FEATURE_NAMES)


def test_pipeline_labels_include_locked_meta_features():
    ticks = make_ticks(n=500, buy_prob=0.9)
    bars, events, labeled, _splits, _state = run_from_ticks(ticks, tight_config())
    assert not labeled.empty
    assert "expected_ticks" in bars.columns
    for name in META_FEATURE_NAMES:
        assert name in labeled.columns
        assert name in events.columns
    assert (labeled["flow_strength"] > 0).all()
    assert (labeled["cusum_excess_ratio"] >= 1.0).all()
    assert (labeled["tick_rel"] > 0).all()
    assert labeled["sigma"].notna().all()
