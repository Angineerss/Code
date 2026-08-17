import pandas as pd

from src.config import PipelineConfig
from src.features import META_FEATURE_NAMES, attach_meta_features, meta_feature_matrix
from src.pipeline import run_from_ticks
from tests.helpers import make_ticks, tight_config


def test_meta_feature_names_locked_set():
    assert META_FEATURE_NAMES == (
        "flow_strength",
        "sigma",
    )
    assert "tick_rel" not in META_FEATURE_NAMES
    assert "tick_count" not in META_FEATURE_NAMES
    assert "cusum_excess_ratio" not in META_FEATURE_NAMES
    assert PipelineConfig().meta_features == META_FEATURE_NAMES
    assert PipelineConfig().event_mode == "every_bar"
    assert PipelineConfig().bar_type == "dollar"
    assert PipelineConfig().require_strong_imbalance is False
    assert PipelineConfig().require_cusum_flow_agree is False
    assert PipelineConfig().pt == 1.0
    assert PipelineConfig().sl == 1.0
    assert PipelineConfig().vertical_bars == 20
    assert PipelineConfig(bar_type="dollar_imbalance").bar_type == "dollar_imbalance"


def test_attach_meta_features_flow_strength_only():
    bars = pd.DataFrame(
        {
            "bar_id": [0, 1],
            "signed_flow": [2_000_000.0, -1_500_000.0],
            "quote_volume": [2_000_000.0, 2_000_000.0],
            "threshold": [1_000_000.0, 1_000_000.0],
            "close_reason": ["dollar", "max_ticks"],
            "tick_count": [100, 500],
            "expected_ticks": [200.0, 250.0],
            "log_ret": [0.01, -0.02],
        }
    )
    events = pd.DataFrame(
        {
            "bar_id": [0, 1],
            "side": [1, -1],
        }
    )
    out = attach_meta_features(bars, events, vol_span=2)
    assert abs(out.loc[0, "flow_strength"] - 1.0) < 1e-9
    assert abs(out.loc[1, "flow_strength"] - 0.75) < 1e-9
    assert "tick_rel" not in out.columns
    assert list(meta_feature_matrix(out).columns) == list(META_FEATURE_NAMES)
    assert "cusum_excess_ratio" not in out.columns


def test_attach_meta_features_falls_back_to_threshold_without_quote():
    bars = pd.DataFrame(
        {
            "bar_id": [0],
            "signed_flow": [2_000_000.0],
            "threshold": [1_000_000.0],
            "log_ret": [0.01],
        }
    )
    events = pd.DataFrame({"bar_id": [0], "side": [1]})
    out = attach_meta_features(bars, events, vol_span=2)
    assert abs(out.loc[0, "flow_strength"] - 2.0) < 1e-9


def test_pipeline_labels_include_locked_meta_features():
    ticks = make_ticks(n=500, buy_prob=0.9)
    bars, events, labeled, _splits, _state = run_from_ticks(ticks, tight_config())
    assert not labeled.empty
    for name in META_FEATURE_NAMES:
        assert name in labeled.columns
        assert name in events.columns
    assert (labeled["flow_strength"] > 0).all()
    assert (labeled["flow_strength"] <= 1.0 + 1e-9).all()
    assert labeled["sigma"].notna().all()
    assert "tick_rel" not in labeled.columns
    assert "cusum_excess_ratio" not in labeled.columns
