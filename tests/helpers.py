from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.config import PipelineConfig


def make_ticks(
    n: int = 400,
    start_price: float = 100.0,
    buy_prob: float = 0.8,
    qty: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    side = np.where(rng.random(n) < buy_prob, 1, -1).astype(np.int8)
    price = start_price + np.cumsum(side * 0.02)
    start = datetime(2024, 1, 15, tzinfo=timezone.utc)
    ts = [start + timedelta(milliseconds=i * 200) for i in range(n)]
    qty_arr = np.full(n, qty, dtype=float)
    return pd.DataFrame(
        {
            "trade_id": np.arange(n),
            "timestamp": pd.to_datetime(ts, utc=True),
            "price": price,
            "qty": qty_arr,
            "side": side,
            "quote_qty": price * qty_arr,
        }
    )


def tight_config(**kwargs) -> PipelineConfig:
    defaults = dict(
        initial_expected_ticks=20,
        imbalance_ewma_span=10,
        cusum_mode="absolute",
        cusum_absolute_h=0.01,
        pt=1.0,
        sl=1.0,
        vertical_bars=5,
        barrier_vol_span=5,
        n_cpcv_groups=4,
        n_cpcv_test_groups=1,
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)
