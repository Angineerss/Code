"""AFML imbalance bars from aggressor-signed ticks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BarType, PipelineConfig


@dataclass(frozen=True)
class _TickArrays:
    ts: np.ndarray
    price: np.ndarray
    qty: np.ndarray
    quote: np.ndarray
    side: np.ndarray


def _ewma_update(prev: float, value: float, alpha: float) -> float:
    return alpha * value + (1.0 - alpha) * prev


def _signed_flow(price: np.ndarray, qty: np.ndarray, side: np.ndarray, bar_type: BarType) -> np.ndarray:
    if bar_type == "tick_imbalance":
        return side.astype(np.float64)
    if bar_type == "volume_imbalance":
        return side.astype(np.float64) * qty
    if bar_type == "dollar_imbalance":
        return side.astype(np.float64) * qty * price
    raise ValueError(f"Unknown bar_type: {bar_type}")


def build_imbalance_bars(ticks: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Sample bars when |signed flow| exceeds the AFML expected-imbalance threshold.

    Expected ticks per bar and expected signed flow per tick are EWMA-updated
    after each closed bar. The first bar is forced closed at
    ``initial_expected_ticks`` so the EWMA has a seed.
    """
    if ticks.empty:
        return _empty_bars()

    arrays = _TickArrays(
        ts=ticks["timestamp"].to_numpy(),
        price=ticks["price"].to_numpy(dtype=np.float64),
        qty=ticks["qty"].to_numpy(dtype=np.float64),
        quote=ticks["quote_qty"].to_numpy(dtype=np.float64),
        side=ticks["side"].to_numpy(dtype=np.int8),
    )
    flow = _signed_flow(arrays.price, arrays.qty, arrays.side, config.bar_type)
    alpha = 2.0 / (config.imbalance_ewma_span + 1.0)

    expected_ticks = float(config.initial_expected_ticks)
    expected_flow_per_tick = np.nan
    theta = 0.0
    start = 0
    n_ticks = 0
    rows: list[list[object]] = []

    for i in range(len(flow)):
        theta += flow[i]
        n_ticks += 1
        warmup = np.isnan(expected_flow_per_tick)
        threshold = expected_ticks if warmup else expected_ticks * abs(expected_flow_per_tick)
        close_bar = (warmup and n_ticks >= config.initial_expected_ticks) or (
            not warmup and abs(theta) >= max(threshold, 1e-12)
        )
        if not close_bar:
            continue

        sl = slice(start, i + 1)
        px = arrays.price[sl]
        qty = arrays.qty[sl]
        quote = arrays.quote[sl]
        side = arrays.side[sl]
        buy = side > 0
        rows.append(
            [
                arrays.ts[start],
                arrays.ts[i],
                px[0],
                px.max(),
                px.min(),
                px[-1],
                qty.sum(),
                quote.sum(),
                n_ticks,
                qty[buy].sum(),
                qty[~buy].sum(),
                theta,
                threshold,
            ]
        )
        expected_ticks = _ewma_update(expected_ticks, float(n_ticks), alpha)
        expected_flow_per_tick = (
            float(theta / n_ticks)
            if np.isnan(expected_flow_per_tick)
            else _ewma_update(expected_flow_per_tick, float(theta / n_ticks), alpha)
        )
        theta = 0.0
        n_ticks = 0
        start = i + 1

    bars = pd.DataFrame(
        rows,
        columns=[
            "start_ts",
            "end_ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "tick_count",
            "buy_volume",
            "sell_volume",
            "signed_flow",
            "threshold",
        ],
    )
    if bars.empty:
        return bars
    bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
    bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
    bars["log_ret"] = np.log(bars["close"]).diff()
    bars["bar_id"] = np.arange(len(bars), dtype=np.int64)
    return bars


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "start_ts",
            "end_ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "tick_count",
            "buy_volume",
            "sell_volume",
            "signed_flow",
            "threshold",
            "log_ret",
            "bar_id",
        ]
    )
