import numpy as np

from app.backtest import run_backtest
from app.config import SimulationConfig
from app.data import NODES, generate_synthetic_market
from app.hawkes import fit_hawkes_network
from app.regimes import detect_regimes


def test_replay_is_reproducible_and_has_events():
    config = SimulationConfig(seed=7, duration_seconds=300)
    first = generate_synthetic_market(config)
    second = generate_synthetic_market(config)

    assert first.nodes == NODES
    assert first.counts.shape == (300, len(NODES))
    assert len(first.events) > 100
    assert np.array_equal(first.counts, second.counts)
    assert np.all(first.prices > 0)


def test_model_and_backtest_return_a_directed_result():
    frame = generate_synthetic_market(SimulationConfig(seed=42, duration_seconds=300))
    train_index = 195
    model = fit_hawkes_network(frame.counts[:train_index], frame.nodes)
    regimes = detect_regimes(frame)
    report = run_backtest(frame, model, regimes, start_index=train_index)

    assert model.excitation.shape == (len(NODES), len(NODES))
    assert np.all(np.diag(model.excitation) == 0.0)
    assert len(model.edges(minimum=0.0)) == len(NODES) * (len(NODES) - 1)
    assert report.summary["trades"] >= 0
    assert report.summary["average_cost_bps"] >= 0

