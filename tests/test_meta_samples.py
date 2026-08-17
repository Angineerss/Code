from datetime import date

import pandas as pd
import pytest

from src.config import PipelineConfig
from src.meta_samples import (
    add_split_column,
    boundary_embargo_mask,
    boundary_purge_mask,
    filter_meta_learning_samples,
)


def _labeled_frame() -> pd.DataFrame:
    # warmup, IS (safe), IS (t1 into OOS), IS (near boundary), OOS
    # bar spans keep seconds/bar ~ 1 hour so 1τ=20-bar purge/embargo ≈ 20h.
    return pd.DataFrame(
        {
            "event_ts": pd.to_datetime(
                [
                    "2018-01-01T00:00:00Z",
                    "2024-06-01T00:00:00Z",
                    "2024-12-20T00:00:00Z",
                    "2024-12-30T00:00:00Z",
                    "2025-01-05T00:00:00Z",
                ],
                utc=True,
            ),
            "t1_ts": pd.to_datetime(
                [
                    "2018-01-01T20:00:00Z",
                    "2024-06-01T20:00:00Z",
                    "2025-01-02T00:00:00Z",  # crosses OOS
                    "2024-12-31T12:00:00Z",  # ends inside IS, near boundary
                    "2025-01-06T00:00:00Z",
                ],
                utc=True,
            ),
            "bar_id": [0, 100, 200, 300, 400],
            "t1_bar_id": [20, 120, 220, 320, 420],
            "y_meta": [0, 1, 1, 0, 1],
        }
    )


def test_filter_drops_warmup_and_oos():
    config = PipelineConfig()
    out = filter_meta_learning_samples(
        _labeled_frame(), config, apply_boundary_purge=False, apply_boundary_embargo=False
    )
    days = out["event_ts"].dt.tz_convert("UTC").dt.date
    assert date(2018, 1, 1) not in set(days)
    assert date(2025, 1, 5) not in set(days)
    assert set(out["split"]) == {"is"}


def test_boundary_purge_drops_t1_into_oos():
    config = PipelineConfig()
    df = add_split_column(_labeled_frame(), config)
    is_rows = df.loc[df["split"] == "is"]
    keep = boundary_purge_mask(is_rows, config)
    kept = is_rows.loc[keep]
    assert date(2024, 12, 20) not in set(kept["event_ts"].dt.date)
    assert date(2024, 6, 1) in set(kept["event_ts"].dt.date)


def test_filter_with_purge_and_embargo_defaults():
    config = PipelineConfig()
    assert config.boundary_purge is True
    assert config.boundary_embargo is True
    # Policy A: Purge + Embargo = 1τ (None → follow vertical_bars=20).
    assert config.purge_bars is None
    assert config.embargo_bars is None
    assert config.resolved_purge_bars() == config.vertical_bars == 20
    assert config.resolved_embargo_bars() == 20
    assert config.selection_method == "cpcv_only"
    out = filter_meta_learning_samples(_labeled_frame(), config)
    # warmup + OOS gone; purged crossing / near-boundary rows gone
    assert date(2024, 12, 20) not in set(out["event_ts"].dt.date)
    assert date(2024, 12, 30) not in set(out["event_ts"].dt.date)
    assert date(2024, 6, 1) in set(out["event_ts"].dt.date)
    assert (out["split"] == "is").all()


def test_selection_method_must_be_cpcv_only():
    with pytest.raises(ValueError, match="cpcv_only"):
        PipelineConfig(selection_method="holdout")
