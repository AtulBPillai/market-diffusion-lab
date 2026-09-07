from fastapi.testclient import TestClient

from app.main import app


def test_api_exposes_a_complete_run():
    with TestClient(app) as client:
        health = client.get("/api/health")
        metadata = client.get("/api/metadata")
        network = client.get("/api/network")
        backtest = client.get("/api/backtest")
        page = client.get("/")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert metadata.status_code == 200
    assert metadata.json()["events"] > 0
    assert network.status_code == 200
    assert "nodes" in network.json() and "edges" in network.json()
    assert backtest.status_code == 200
    assert "summary" in backtest.json()
    assert page.status_code == 200
    assert "Market Diffusion Lab" in page.text

