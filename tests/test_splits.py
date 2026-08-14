from datetime import date, timedelta

import pytest

from src.config import PipelineConfig


def test_is_oos_cut_is_contiguous_and_covers_universe():
    config = PipelineConfig()
    assert config.universe_start == date(2024, 1, 1)
    assert config.is_end == date(2025, 12, 31)
    assert config.oos_start == date(2026, 1, 1)
    assert config.oos_end == date(2026, 8, 13)
    assert config.oos_start == config.is_end + timedelta(days=1)
    is_days = (config.is_end - config.universe_start).days + 1
    oos_days = (config.oos_end - config.oos_start).days + 1
    assert is_days == 731
    assert oos_days == 225
    assert is_days + oos_days == (config.oos_end - config.universe_start).days + 1


def test_split_for_day():
    config = PipelineConfig()
    assert config.split_for_day(date(2023, 12, 31)) == "out_of_universe"
    assert config.split_for_day(date(2024, 1, 1)) == "is"
    assert config.split_for_day(date(2025, 12, 31)) == "is"
    assert config.split_for_day(date(2026, 1, 1)) == "oos"
    assert config.split_for_day(date(2026, 8, 13)) == "oos"
    assert config.split_for_day(date(2026, 8, 14)) == "out_of_universe"


def test_oos_must_follow_is():
    with pytest.raises(ValueError, match="OOS must start"):
        PipelineConfig(is_end=date(2025, 12, 31), oos_start=date(2026, 1, 2))
