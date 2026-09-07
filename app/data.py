"""Synthetic, reproducible multi-venue market event generation.

The simulator deliberately uses a stable multivariate Hawkes-style process. It is
not intended to represent live exchange data or an executable trading signal; it
creates a controlled environment in which the full modelling pipeline can be
tested without credentials or paid market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SimulationConfig


NODES: tuple[str, ...] = (
    "BINANCE:BTCUSDT",
    "COINBASE:BTCUSD",
    "KRAKEN:BTCUSD",
    "BINANCE:ETHUSDT",
)


@dataclass(frozen=True)
class MarketEvent:
    """One normalized trade event used by the replay pipeline."""

    timestamp_ms: int
    node: str
    side: int
    volume: float
    mid_price: float
    spread_bps: float
    imbalance: float

    def to_dict(self) -> dict[str, Any]:
        venue, symbol = self.node.split(":", maxsplit=1)
        return {
            "timestamp_ms": self.timestamp_ms,
            "venue": venue,
            "symbol": symbol,
            "node": self.node,
            "side": "buy" if self.side > 0 else "sell",
            "volume": round(self.volume, 5),
            "mid_price": round(self.mid_price, 4),
            "spread_bps": round(self.spread_bps, 3),
            "imbalance": round(self.imbalance, 4),
        }


@dataclass
class MarketFrame:
    """Binned event tensors plus the raw normalized events."""

    nodes: tuple[str, ...]
    timestamps_ms: np.ndarray
    counts: np.ndarray
    signed_volume: np.ndarray
    prices: np.ndarray
    spreads_bps: np.ndarray
    imbalance: np.ndarray
    events: list[MarketEvent]
    true_regimes: np.ndarray
    bin_ms: int

    def preview_events(self, limit: int = 30) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events[:limit]]


def _regime_for_fraction(fraction: float) -> tuple[str, float]:
    """Return the latent regime and its intensity multiplier."""

    if 0.30 <= fraction < 0.43:
        return "volatile", 1.75
    if 0.43 <= fraction < 0.55:
        return "dislocated", 2.25
    if 0.76 <= fraction < 0.84:
        return "volatile", 1.55
    if 0.84 <= fraction < 0.90:
        return "dislocated", 2.35
    return "calm", 1.0


def _true_excitation() -> np.ndarray:
    """Known source-to-target excitation used only by the simulator.

    The first BTC venue is intentionally the main price-discovery leader. The
    estimator never receives this matrix; it must recover the relationship from
    timestamped event counts.
    """

    return np.array(
        [
            [0.19, 0.34, 0.24, 0.17],
            [0.03, 0.16, 0.13, 0.06],
            [0.02, 0.08, 0.14, 0.04],
            [0.04, 0.10, 0.06, 0.15],
        ],
        dtype=float,
    )


def generate_synthetic_market(config: SimulationConfig) -> MarketFrame:
    """Generate an event replay with latent calm, volatile, and dislocated regimes.

    Event intensity follows a discrete-time exponential-kernel Hawkes process:
    lambda_j(t) = mu_j(t) + sum_i alpha_ij * H_i(t).
    Directional order flow and price changes additionally follow the same causal
    graph, which makes the out-of-sample execution test meaningful.
    """

    rng = np.random.default_rng(config.seed)
    n_nodes = len(NODES)
    n_bins = config.duration_seconds * 1_000 // config.bin_ms
    timestamps = np.arange(n_bins, dtype=np.int64) * config.bin_ms

    counts = np.zeros((n_bins, n_nodes), dtype=float)
    signed_volume = np.zeros((n_bins, n_nodes), dtype=float)
    prices = np.zeros((n_bins, n_nodes), dtype=float)
    spreads = np.zeros((n_bins, n_nodes), dtype=float)
    imbalance = np.zeros((n_bins, n_nodes), dtype=float)
    regimes = np.empty(n_bins, dtype="U16")

    base_rates = np.array([0.17, 0.13, 0.11, 0.13], dtype=float)
    base_prices = np.array([65_000.0, 65_005.0, 65_001.0, 3_450.0], dtype=float)
    price_now = base_prices.copy()
    excitation = _true_excitation()
    kernel = np.zeros(n_nodes, dtype=float)
    directional_state = np.zeros(n_nodes, dtype=float)
    prior_directional_state = np.zeros(n_nodes, dtype=float)
    prior_signed_volume = np.zeros(n_nodes, dtype=float)
    events: list[MarketEvent] = []

    # News bursts are exogenous impulses. The model must infer their diffusion
    # from market events rather than accessing this hidden variable.
    shock_bins = {int(n_bins * p) for p in (0.34, 0.47, 0.80)}

    for t in range(n_bins):
        fraction = t / max(n_bins - 1, 1)
        regime, regime_multiplier = _regime_for_fraction(fraction)
        regimes[t] = regime
        global_shock = rng.normal(0.0, 0.55)
        if t in shock_bins:
            global_shock += rng.choice([-1.0, 1.0]) * rng.uniform(3.2, 4.8)

        # The leader receives the strongest information impulse; followers get
        # the previous bin's source signal through the directed graph.
        propagated = prior_directional_state @ excitation
        leader_bias = np.array([1.25, 0.30, 0.18, 0.24]) * global_shock
        directional_state = (
            0.48 * directional_state
            + 0.46 * propagated
            + leader_bias
            + rng.normal(0.0, 0.72 * regime_multiplier, size=n_nodes)
        )

        intensity = base_rates * regime_multiplier + kernel @ excitation
        intensity = np.clip(intensity, 0.015, 2.4)
        event_counts = np.minimum(rng.poisson(intensity), 6)
        counts[t] = event_counts

        for i, event_count in enumerate(event_counts):
            if event_count == 0:
                continue
            local_spread = (
                1.25
                + 0.38 * i
                + 0.42 * regime_multiplier
                + abs(directional_state[i]) * 0.09
            )
            for _ in range(int(event_count)):
                side_score = directional_state[i] + rng.normal(0.0, 0.85)
                side = 1 if side_score >= 0 else -1
                volume = float(rng.lognormal(mean=2.9, sigma=0.42))
                signed_volume[t, i] += side * volume
                event_timestamp = int(timestamps[t] + rng.integers(0, config.bin_ms))
                events.append(
                    MarketEvent(
                        timestamp_ms=event_timestamp,
                        node=NODES[i],
                        side=side,
                        volume=volume,
                        mid_price=float(price_now[i]),
                        spread_bps=float(local_spread),
                        imbalance=0.0,
                    )
                )

        imbalance[t] = np.tanh(signed_volume[t] / 90.0)
        idiosyncratic = rng.normal(0.0, 0.000055 * regime_multiplier, size=n_nodes)
        flow_return = 0.000010 * signed_volume[t]
        # A source trade affects followers on the next bin. This explicit lag is
        # what makes a timestamp-correct lead–lag execution test possible.
        propagated_return = 0.000032 * (prior_signed_volume @ excitation)
        market_return = (
            flow_return
            + propagated_return
            + idiosyncratic
            + np.array([0.000045, 0.000028, 0.000023, 0.000035]) * global_shock
        )
        if regime == "dislocated":
            # Fragmented liquidity raises adverse-selection risk and makes the
            # no-regime-filter baseline materially worse.
            market_return += rng.normal(0.0, 0.00036, size=n_nodes)
        price_now = np.maximum(1.0, price_now * np.exp(market_return))
        prices[t] = price_now
        spreads[t] = 1.1 + 0.25 * np.arange(n_nodes) + 0.43 * regime_multiplier
        kernel = kernel * config.hawkes_decay + event_counts
        prior_directional_state = directional_state.copy()
        prior_signed_volume = signed_volume[t].copy()

    # Events were appended by bin but their jittered timestamps must be sorted for
    # an exchange-style replay and for downstream CSV export.
    events.sort(key=lambda event: event.timestamp_ms)
    return MarketFrame(
        nodes=NODES,
        timestamps_ms=timestamps,
        counts=counts,
        signed_volume=signed_volume,
        prices=prices,
        spreads_bps=spreads,
        imbalance=imbalance,
        events=events,
        true_regimes=regimes,
        bin_ms=config.bin_ms,
    )
