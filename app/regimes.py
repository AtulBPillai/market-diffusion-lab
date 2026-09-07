"""Online-style market regime detection from prices and order flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import MarketFrame


@dataclass
class RegimeResult:
    """Detected state for each replay bin."""

    labels: np.ndarray
    stress_score: np.ndarray
    volatility_bps: np.ndarray
    dispersion_bps: np.ndarray

    def payload(self, timestamps_ms: np.ndarray, max_points: int = 180) -> list[dict[str, Any]]:
        stride = max(1, len(timestamps_ms) // max_points)
        return [
            {
                "timestamp_ms": int(timestamps_ms[index]),
                "regime": str(self.labels[index]),
                "stress_score": round(float(self.stress_score[index]), 3),
                "volatility_bps": round(float(self.volatility_bps[index]), 3),
                "dispersion_bps": round(float(self.dispersion_bps[index]), 3),
            }
            for index in range(0, len(timestamps_ms), stride)
        ]


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean with a valid value from the first sample."""

    output = np.empty_like(values, dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    for index in range(len(values)):
        start = max(0, index - window + 1)
        output[index] = (cumulative[index + 1] - cumulative[start]) / (index - start + 1)
    return output


def detect_regimes(frame: MarketFrame, window: int = 30) -> RegimeResult:
    """Classify calm, volatile, and dislocated conditions without future data.

    The detector uses realized price movement, cross-venue BTC dispersion, and
    order-flow pressure. Every feature at t depends only on observations through
    t, so it is safe for the out-of-sample backtest.
    """

    log_prices = np.log(np.maximum(frame.prices, 1e-9))
    returns = np.vstack([np.zeros((1, frame.prices.shape[1])), np.diff(log_prices, axis=0)])
    average_abs_return = np.mean(np.abs(returns), axis=1) * 10_000
    volatility_bps = _rolling_mean(average_abs_return, window)

    btc_prices = frame.prices[:, :3]
    btc_mean = np.mean(btc_prices, axis=1)
    dispersion_bps = np.ptp(btc_prices, axis=1) / np.maximum(btc_mean, 1e-9) * 10_000
    flow_pressure = _rolling_mean(np.mean(np.abs(frame.imbalance), axis=1), window)

    warm_start = min(len(volatility_bps) - 1, window * 2)
    baseline_vol = max(0.15, float(np.median(volatility_bps[: warm_start + 1])))
    baseline_dispersion = max(0.10, float(np.median(dispersion_bps[: warm_start + 1])))
    stress_score = (
        0.58 * volatility_bps / baseline_vol
        + 0.30 * dispersion_bps / baseline_dispersion
        + 0.12 * flow_pressure / max(0.03, float(np.median(flow_pressure[: warm_start + 1])))
    )

    labels = np.full(len(stress_score), "calm", dtype="U16")
    dislocation_cutoff = max(3.0, float(np.quantile(dispersion_bps, 0.84)))
    volatile_cutoff = max(1.7, float(np.quantile(stress_score, 0.68)))
    labels[stress_score >= volatile_cutoff] = "volatile"
    labels[dispersion_bps >= dislocation_cutoff] = "dislocated"
    labels[: min(window, len(labels))] = "calm"
    return RegimeResult(
        labels=labels,
        stress_score=stress_score,
        volatility_bps=volatility_bps,
        dispersion_bps=dispersion_bps,
    )

