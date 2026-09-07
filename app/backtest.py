"""Execution-cost-aware out-of-sample lead-lag backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import MarketFrame
from .hawkes import HawkesFitResult
from .regimes import RegimeResult


@dataclass(frozen=True)
class Trade:
    timestamp_ms: int
    source: str
    target: str
    side: str
    excitation: float
    gross_bps: float
    cost_bps: float
    net_pnl_usd: float
    regime: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "source": self.source,
            "target": self.target,
            "side": self.side,
            "excitation": round(self.excitation, 4),
            "gross_bps": round(self.gross_bps, 3),
            "cost_bps": round(self.cost_bps, 3),
            "net_pnl_usd": round(self.net_pnl_usd, 2),
            "regime": self.regime,
        }


@dataclass
class BacktestReport:
    trades: list[Trade]
    equity_curve: list[dict[str, float | int]]
    summary: dict[str, float | int]
    baseline_summary: dict[str, float | int]

    def payload(self, max_trades: int = 80) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "baseline_summary": self.baseline_summary,
            "equity_curve": self.equity_curve,
            "trades": [trade.to_dict() for trade in self.trades[-max_trades:]],
        }


def _summarize(trades: list[Trade], initial_capital: float = 100_000.0) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    pnl = np.array([trade.net_pnl_usd for trade in trades], dtype=float)
    equity = initial_capital + np.cumsum(pnl) if len(pnl) else np.array([initial_capital])
    running_peak = np.maximum.accumulate(equity)
    drawdown = (equity - running_peak) / running_peak * 100.0
    sharpe = 0.0
    if len(pnl) >= 2 and float(np.std(pnl, ddof=1)) > 1e-9:
        sharpe = float(np.mean(pnl) / np.std(pnl, ddof=1) * np.sqrt(len(pnl)))
    win_rate = float(np.mean(pnl > 0) * 100.0) if len(pnl) else 0.0
    average_cost = float(np.mean([trade.cost_bps for trade in trades])) if trades else 0.0
    summary: dict[str, float | int] = {
        "trades": int(len(trades)),
        "net_pnl_usd": round(float(np.sum(pnl)), 2),
        "win_rate_pct": round(win_rate, 2),
        "sharpe_like": round(sharpe, 3),
        "max_drawdown_pct": round(float(np.min(drawdown)), 3),
        "average_cost_bps": round(average_cost, 3),
    }
    curve: list[dict[str, float | int]] = []
    if trades:
        stride = max(1, len(trades) // 180)
        for index in range(0, len(trades), stride):
            curve.append(
                {
                    "timestamp_ms": trades[index].timestamp_ms,
                    "equity": round(float(equity[index]), 2),
                }
            )
        if curve[-1]["timestamp_ms"] != trades[-1].timestamp_ms:
            curve.append({"timestamp_ms": trades[-1].timestamp_ms, "equity": round(float(equity[-1]), 2)})
    else:
        curve.append({"timestamp_ms": 0, "equity": initial_capital})
    return summary, curve


def _simulate_strategy(
    frame: MarketFrame,
    model: HawkesFitResult,
    regimes: RegimeResult,
    *,
    start_index: int,
    respect_regime: bool,
    top_edges: int,
    horizon: int = 2,
) -> list[Trade]:
    """Generate cost-aware trades using only preceding source order flow."""

    edges = model.leading_edges(limit=top_edges)
    trades: list[Trade] = []
    last_entry: dict[tuple[int, int], int] = {}
    notional = 10_000.0

    for source, target, excitation in edges:
        source_flow = frame.signed_volume[:, source]
        flow_threshold = max(10.0, float(np.quantile(np.abs(source_flow[:start_index]), 0.73)))
        for t in range(max(start_index, 3), len(frame.timestamps_ms) - horizon):
            key = (source, target)
            if t - last_entry.get(key, -99) < 4:
                continue
            regime = str(regimes.labels[t])
            if respect_regime and regime == "dislocated":
                continue
            flow_signal = float(np.mean(source_flow[t - 2 : t + 1]))
            if abs(flow_signal) < flow_threshold:
                continue

            side = 1 if flow_signal > 0 else -1
            raw_return = float(np.log(frame.prices[t + horizon, target] / frame.prices[t, target]) * 10_000)
            gross_bps = side * raw_return
            # Conservative all-in round-trip cost: taker fee, half-spread twice,
            # and slippage that grows in stressed regimes.
            half_spread = float(np.mean(frame.spreads_bps[t : t + 1, target])) / 2.0
            if regime == "dislocated":
                stress_slippage = 8.5
            elif regime == "volatile":
                stress_slippage = 2.25
            else:
                stress_slippage = 1.0
            cost_bps = 1.9 + (2.0 * half_spread) + stress_slippage
            net_pnl = notional * (gross_bps - cost_bps) / 10_000.0
            trades.append(
                Trade(
                    timestamp_ms=int(frame.timestamps_ms[t]),
                    source=frame.nodes[source],
                    target=frame.nodes[target],
                    side="long" if side > 0 else "short",
                    excitation=excitation,
                    gross_bps=gross_bps,
                    cost_bps=cost_bps,
                    net_pnl_usd=net_pnl,
                    regime=regime,
                )
            )
            last_entry[key] = t

    return sorted(trades, key=lambda trade: trade.timestamp_ms)


def run_backtest(
    frame: MarketFrame,
    model: HawkesFitResult,
    regimes: RegimeResult,
    *,
    start_index: int,
) -> BacktestReport:
    """Evaluate regime-aware diffusion signals against a naive static baseline."""

    trades = _simulate_strategy(
        frame,
        model,
        regimes,
        start_index=start_index,
        respect_regime=True,
        top_edges=3,
    )
    baseline_trades = _simulate_strategy(
        frame,
        model,
        regimes,
        start_index=start_index,
        respect_regime=False,
        top_edges=3,
    )
    summary, equity_curve = _summarize(trades)
    baseline_summary, _ = _summarize(baseline_trades)
    return BacktestReport(
        trades=trades,
        equity_curve=equity_curve,
        summary=summary,
        baseline_summary=baseline_summary,
    )
