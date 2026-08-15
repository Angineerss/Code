from datetime import date, timedelta

import pytest

from src.config import PipelineConfig


def test_is_oos_cut_uses_full_archive_with_warmup():
    config = PipelineConfig()
    assert config.archive_start == date(2017, 8, 17)
    assert config.warmup_end == date(2018, 8, 16)
    assert config.universe_start == date(2018, 8, 17)
    assert config.is_end == date(2024, 12, 31)
    assert config.oos_start == date(2025, 1, 1)
    assert config.oos_end == date(2026, 8, 13)
    assert config.universe_start == config.warmup_end + timedelta(days=1)
    assert config.oos_start == config.is_end + timedelta(days=1)

    warmup_days = (config.warmup_end - config.archive_start).days + 1
    is_days = (config.is_end - config.universe_start).days + 1
    oos_days = (config.oos_end - config.oos_start).days + 1
    assert warmup_days == 365
    assert is_days == 2329
    assert oos_days == 590
    assert warmup_days + is_days + oos_days == (config.oos_end - config.archive_start).days + 1


def test_split_for_day_warmup_is_oos():
    config = PipelineConfig()
    assert config.split_for_day(date(2017, 8, 16)) == "out_of_universe"
    assert config.split_for_day(date(2017, 8, 17)) == "warmup"
    assert config.split_for_day(date(2018, 8, 16)) == "warmup"
    assert config.split_for_day(date(2018, 8, 17)) == "is"
    assert config.split_for_day(date(2024, 12, 31)) == "is"
    assert config.split_for_day(date(2025, 1, 1)) == "oos"
    assert config.split_for_day(date(2026, 8, 13)) == "oos"
    assert config.split_for_day(date(2026, 8, 14)) == "out_of_universe"


def test_is_cpcv_is_train_vs_cv_only_inside_is():
    """IS holds both learning and CV via CPCV; OOS is never used for CV."""
    config = PipelineConfig()
    assert config.cv_method == "cpcv"
    assert config.n_cpcv_groups == 6
    assert config.n_cpcv_test_groups == 2
    assert config.cpcv_path_count() == 15  # C(6,2)
    assert config.is_range() == (date(2018, 8, 17), date(2024, 12, 31))
    assert config.oos_range() == (date(2025, 1, 1), date(2026, 8, 13))
    assert config.warmup_range() == (date(2017, 8, 17), date(2018, 8, 16))


def test_oos_must_follow_is():
    with pytest.raises(ValueError, match="OOS must start"):
        PipelineConfig(is_end=date(2024, 12, 31), oos_start=date(2025, 1, 2))


def test_is_must_follow_warmup():
    with pytest.raises(ValueError, match="IS must start"):
        PipelineConfig(warmup_end=date(2018, 8, 16), universe_start=date(2018, 8, 18))
