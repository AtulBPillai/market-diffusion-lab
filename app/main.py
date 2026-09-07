"""FastAPI entry point for Market Diffusion Lab."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import APP_NAME, APP_VERSION, WEB_DIR, SimulationConfig
from .schemas import RunRequest
from .service import AnalysisService


service = AnalysisService()
web_root = Path(WEB_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.rebuild(SimulationConfig())
    yield


app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=web_root), name="assets")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(web_root / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": APP_VERSION}


@app.get("/api/metadata")
def metadata() -> dict:
    return service.metadata()


@app.get("/api/network")
def network(minimum: float = Query(default=0.025, ge=0.0, le=2.0)) -> dict:
    return service.network_payload(minimum)


@app.get("/api/regimes")
def regimes() -> list[dict]:
    return service.regime_payload()


@app.get("/api/backtest")
def backtest() -> dict:
    return service.backtest_payload()


@app.get("/api/events")
def events(limit: int = Query(default=30, ge=1, le=100)) -> list[dict]:
    return service.event_payload(limit)


@app.post("/api/run")
def run(request: RunRequest) -> dict:
    try:
        config = SimulationConfig(
            seed=request.seed,
            duration_seconds=request.duration_seconds,
            bin_ms=request.bin_ms,
        )
        service.rebuild(config)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"status": "completed", "metadata": service.metadata()}

