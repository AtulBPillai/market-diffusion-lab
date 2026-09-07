"""Sparse exponential-kernel Hawkes-style network estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HawkesFitResult:
    """Fitted directed excitation graph and training diagnostics."""

    nodes: tuple[str, ...]
    baseline_rate: np.ndarray
    excitation: np.ndarray  # source x target
    decay: float
    train_loss: list[float]

    def edges(self, minimum: float = 0.025) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for target in range(len(self.nodes)):
            total = float(np.sum(self.excitation[:, target])) + 1e-9
            for source in range(len(self.nodes)):
                if source == target:
                    continue
                value = float(self.excitation[source, target])
                if value >= minimum:
                    rows.append(
                        {
                            "source": self.nodes[source],
                            "target": self.nodes[target],
                            "excitation": round(value, 4),
                            "confidence": round(min(0.99, value / total), 4),
                        }
                    )
        return sorted(rows, key=lambda row: row["excitation"], reverse=True)

    def leading_edges(self, limit: int = 3) -> list[tuple[int, int, float]]:
        candidates: list[tuple[int, int, float]] = []
        for source in range(len(self.nodes)):
            for target in range(len(self.nodes)):
                if source != target:
                    candidates.append((source, target, float(self.excitation[source, target])))
        return sorted(candidates, key=lambda item: item[2], reverse=True)[:limit]


def exponential_history(counts: np.ndarray, decay: float) -> np.ndarray:
    """Compute H_i(t), using only events strictly before t."""

    history = np.zeros_like(counts, dtype=float)
    state = np.zeros(counts.shape[1], dtype=float)
    for t in range(counts.shape[0]):
        history[t] = state
        state = decay * state + counts[t]
    return history


def fit_hawkes_network(
    counts: np.ndarray,
    nodes: tuple[str, ...],
    *,
    decay: float = 0.65,
    iterations: int = 1_100,
    learning_rate: float = 0.08,
    l1_penalty: float = 0.0015,
    l2_penalty: float = 0.001,
) -> HawkesFitResult:
    """Fit a sparse discrete-time exponential-kernel Hawkes approximation.

    For each target venue j, this optimizes the Hawkes conditional Poisson
    likelihood lambda_j(t) = b_j + sum_i a_ij H_i(t), subject to non-negative
    rate and excitation constraints. The a_ij coefficients are the directed,
    interpretable excitation strengths displayed in the UI.
    """

    if counts.ndim != 2 or counts.shape[1] != len(nodes):
        raise ValueError("counts must be a T x N array matching nodes")
    if counts.shape[0] < 100:
        raise ValueError("at least 100 bins are required to fit the model")

    history = exponential_history(counts, decay)
    n_bins, n_nodes = counts.shape
    X = np.column_stack([np.ones(n_bins), history])
    coefficients = np.zeros((n_nodes, n_nodes + 1), dtype=float)
    losses: list[float] = []

    for target in range(n_nodes):
        y = counts[:, target]
        theta = np.full(n_nodes + 1, 0.01, dtype=float)
        theta[0] = max(0.002, float(np.mean(y) * 0.82))
        target_losses: list[float] = []

        for step in range(iterations):
            intensity = np.clip(X @ theta, 1e-6, 12.0)
            current_loss = float(
                np.mean(intensity - y * np.log(intensity))
                + l1_penalty * np.sum(theta[1:])
                + 0.5 * l2_penalty * np.sum(theta[1:] ** 2)
            )
            gradient = (X.T @ (1.0 - y / intensity)) / n_bins
            gradient[1:] += l2_penalty * theta[1:]
            # Backtracking keeps projected gradient descent stable even during a
            # shock-heavy replay. It also makes fitting deterministic by seed.
            step_size = learning_rate
            for _ in range(12):
                candidate = theta - step_size * gradient
                candidate[0] = max(1e-6, candidate[0])
                candidate[1:] = np.maximum(0.0, candidate[1:] - step_size * l1_penalty)
                candidate_intensity = np.clip(X @ candidate, 1e-6, 12.0)
                candidate_loss = float(
                    np.mean(candidate_intensity - y * np.log(candidate_intensity))
                    + l1_penalty * np.sum(candidate[1:])
                    + 0.5 * l2_penalty * np.sum(candidate[1:] ** 2)
                )
                if candidate_loss <= current_loss:
                    theta = candidate
                    break
                step_size *= 0.5

            if step % 100 == 0 or step == iterations - 1:
                fitted_intensity = np.clip(X @ theta, 1e-6, 12.0)
                target_losses.append(
                    float(
                        np.mean(fitted_intensity - y * np.log(fitted_intensity))
                        + l1_penalty * np.sum(theta[1:])
                        + 0.5 * l2_penalty * np.sum(theta[1:] ** 2)
                    )
                )

        coefficients[target] = theta
        losses.append(target_losses[-1])

    excitation = coefficients[:, 1:].T
    np.fill_diagonal(excitation, 0.0)
    return HawkesFitResult(
        nodes=nodes,
        baseline_rate=coefficients[:, 0],
        excitation=excitation,
        decay=decay,
        train_loss=losses,
    )
