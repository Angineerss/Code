"""AFML imbalance bars from aggressor-signed ticks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BarType, PipelineConfig

BAR_COLUMNS = [
    "start_ts",
    "end_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "tick_count",
    "buy_ticks",
    "buy_volume",
    "sell_volume",
    "signed_flow",
    "threshold",
    "close_reason",
]


@dataclass(frozen=True)
class ImbalanceSeed:
    """Prior-year scale for dollar imbalance bars.

    ``expected_imbalance`` is D = mean(prior 1y daily quote notional) / 50.
    """

    expected_imbalance: float
    init_t: int
    expected_size: float | None = None


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


def _clip_imbalance_frac(raw: float, config: PipelineConfig) -> float:
    return float(min(max(raw, config.min_abs_2p1), config.max_abs_2p1))


def _tick_arrays(ticks: pd.DataFrame) -> _TickArrays:
    return _TickArrays(
        ts=ticks["timestamp"].to_numpy(),
        price=ticks["price"].to_numpy(dtype=np.float64),
        qty=ticks["qty"].to_numpy(dtype=np.float64),
        quote=ticks["quote_qty"].to_numpy(dtype=np.float64),
        side=ticks["side"].to_numpy(dtype=np.int8),
    )


def _finalize_bars(rows: list[list[object]]) -> pd.DataFrame:
    bars = pd.DataFrame(rows, columns=BAR_COLUMNS)
    if bars.empty:
        return bars
    bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
    bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
    bars["log_ret"] = np.log(bars["close"]).diff()
    bars["bar_id"] = np.arange(len(bars), dtype=np.int64)
    bars["duration_s"] = (bars["end_ts"] - bars["start_ts"]).dt.total_seconds()
    return bars


def _bar_row(
    arrays: _TickArrays,
    start: int,
    end: int,
    n_ticks: int,
    buy_ticks: int,
    theta: float,
    threshold: float,
    reason: str,
) -> list[object]:
    sl = slice(start, end)
    px = arrays.price[sl]
    qty = arrays.qty[sl]
    quote = arrays.quote[sl]
    side = arrays.side[sl]
    buy = side > 0
    return [
        arrays.ts[start],
        arrays.ts[end - 1],
        px[0],
        px.max(),
        px.min(),
        px[-1],
        qty.sum(),
        quote.sum(),
        n_ticks,
        buy_ticks,
        qty[buy].sum(),
        qty[~buy].sum(),
        theta,
        threshold,
        reason,
    ]


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=BAR_COLUMNS + ["log_ret", "bar_id", "duration_s"])


def daily_quote_volume(ticks: pd.DataFrame) -> float:
    return float(ticks["quote_qty"].sum()) if not ticks.empty else 0.0


def build_imbalance_bars(
    ticks: pd.DataFrame,
    config: PipelineConfig,
    seed: ImbalanceSeed | None = None,
) -> pd.DataFrame:
    """Sample bars when |signed flow| exceeds expected imbalance.

    Expected imbalance follows AFML: ``E[T] * |2P[buy]-1| * E[size]``.
    For dollar imbalance, E[θ] is seeded at D = prior-year average daily
    notional / 50 and clipped around that scale so it cannot run away.
    """
    if ticks.empty:
        return _empty_bars()

    arrays = _tick_arrays(ticks)
    flow = _signed_flow(arrays.price, arrays.qty, arrays.side, config.bar_type)
    abs_flow = np.abs(flow)
    alpha = 2.0 / (config.imbalance_ewma_span + 1.0)

    init_t = int(seed.init_t) if seed is not None else int(config.initial_expected_ticks)
    expected_ticks = float(init_t)
    expected_size = (
        float(seed.expected_size)
        if seed is not None and seed.expected_size is not None
        else np.nan
    )
    expected_2p1 = np.nan
    expected_imbalance = float(seed.expected_imbalance) if seed is not None else np.nan
    theta = 0.0
    start = 0
    n_ticks = 0
    buy_ticks = 0
    size_sum = 0.0
    warmed = False
    rows: list[list[object]] = []

    def threshold_now() -> float:
        if not warmed:
            return float("inf")
        if seed is not None and np.isfinite(expected_imbalance):
            return float(expected_imbalance)
        frac = _clip_imbalance_frac(expected_2p1, config)
        return expected_ticks * frac * expected_size

    for i in range(len(flow)):
        theta += flow[i]
        n_ticks += 1
        buy_ticks += int(arrays.side[i] > 0)
        size_sum += abs_flow[i]

        max_ticks = max(int(expected_ticks * config.max_ticks_mult), init_t)
        if not warmed:
            close_reason = "warmup" if n_ticks >= init_t else None
        elif abs(theta) >= max(threshold_now(), 1e-12):
            close_reason = "imbalance"
        elif n_ticks >= max_ticks:
            close_reason = "max_ticks"
        else:
            close_reason = None
        if close_reason is None:
            continue

        mean_size = size_sum / n_ticks
        raw_2p1 = abs(2.0 * (buy_ticks / n_ticks) - 1.0)
        rows.append(
            _bar_row(
                arrays,
                start,
                i + 1,
                n_ticks,
                buy_ticks,
                theta,
                threshold_now() if warmed else float(n_ticks),
                close_reason,
            )
        )
        if close_reason != "max_ticks":
            expected_ticks = _ewma_update(expected_ticks, float(n_ticks), alpha)
            expected_ticks = min(
                max(expected_ticks, init_t * config.expected_ticks_min_mult),
                init_t * config.expected_ticks_max_mult,
            )
            expected_size = mean_size if np.isnan(expected_size) else _ewma_update(expected_size, mean_size, alpha)
            expected_2p1 = raw_2p1 if np.isnan(expected_2p1) else _ewma_update(expected_2p1, raw_2p1, alpha)
            if seed is not None:
                afml = expected_ticks * _clip_imbalance_frac(expected_2p1, config) * expected_size
                expected_imbalance = _ewma_update(expected_imbalance, float(afml), alpha)
                expected_imbalance = min(
                    max(expected_imbalance, seed.expected_imbalance * config.expected_imbalance_min_mult),
                    seed.expected_imbalance * config.expected_imbalance_max_mult,
                )
        warmed = True
        theta = 0.0
        n_ticks = 0
        buy_ticks = 0
        size_sum = 0.0
        start = i + 1

    return _finalize_bars(rows)


def build_bars(
    ticks: pd.DataFrame,
    config: PipelineConfig,
    seed: ImbalanceSeed | None = None,
) -> pd.DataFrame:
    return build_imbalance_bars(ticks, config, seed=seed)
