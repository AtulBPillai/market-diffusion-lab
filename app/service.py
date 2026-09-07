"""Thread-safe orchestration layer for the API and dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

import numpy as np

from .backtest import BacktestReport, run_backtest
from .config import SimulationConfig
from .data import MarketFrame, generate_synthetic_market
from .hawkes import HawkesFitResult, fit_hawkes_network
from .regimes import RegimeResult, detect_regimes


@dataclass
class AnalysisRun:
    config: SimulationConfig
    frame: MarketFrame
    model: HawkesFitResult
    regimes: RegimeResult
    backtest: BacktestReport
    train_bins: int


class AnalysisService:
    """Owns the most recent reproducible analysis run."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._run: AnalysisRun | None = None

    def rebuild(self, config: SimulationConfig) -> AnalysisRun:
        frame = generate_synthetic_market(config)
        train_bins = int(len(frame.timestamps_ms) * config.train_fraction)
        model = fit_hawkes_network(
            frame.counts[:train_bins],
            frame.nodes,
            decay=config.hawkes_decay,
        )
        regimes = detect_regimes(frame)
        report = run_backtest(frame, model, regimes, start_index=train_bins)
        run = AnalysisRun(config, frame, model, regimes, report, train_bins)
        with self._lock:
            self._run = run
        return run

    def current(self) -> AnalysisRun:
        with self._lock:
            if self._run is None:
                raise RuntimeError("analysis is not initialized")
            return self._run

    def metadata(self) -> dict[str, Any]:
        run = self.current()
        return {
            "source": "reproducible synthetic multi-venue event replay",
            "seed": run.config.seed,
            "duration_seconds": run.config.duration_seconds,
            "bin_ms": run.config.bin_ms,
            "events": len(run.frame.events),
            "nodes": list(run.frame.nodes),
            "train_bins": run.train_bins,
            "test_bins": len(run.frame.timestamps_ms) - run.train_bins,
            "model": "Sparse exponential-kernel Hawkes approximation",
        }

    def network_payload(self, minimum: float = 0.025) -> dict[str, Any]:
        run = self.current()
        node_payload = []
        for index, node in enumerate(run.frame.nodes):
            node_payload.append(
                {
                    "id": node,
                    "latest_price": round(float(run.frame.prices[-1, index]), 4),
                    "events": int(np.sum(run.frame.counts[:, index])),
                    "last_imbalance": round(float(run.frame.imbalance[-1, index]), 4),
                }
            )
        return {
            "nodes": node_payload,
            "edges": run.model.edges(minimum=minimum),
            "fit_loss": [round(float(value), 4) for value in run.model.train_loss],
        }

    def regime_payload(self) -> list[dict[str, Any]]:
        run = self.current()
        return run.regimes.payload(run.frame.timestamps_ms)

    def backtest_payload(self) -> dict[str, Any]:
        return self.current().backtest.payload()

    def event_payload(self, limit: int = 30) -> list[dict[str, Any]]:
        return self.current().frame.preview_events(limit)

