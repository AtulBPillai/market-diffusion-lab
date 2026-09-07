"""Runtime settings for Market Diffusion Lab."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the reproducible market-event simulator."""

    seed: int = 42
    duration_seconds: int = 1_200
    bin_ms: int = 1_000
    hawkes_decay: float = 0.65
    train_fraction: float = 0.65

    def __post_init__(self) -> None:
        if self.duration_seconds < 300:
            raise ValueError("duration_seconds must be at least 300")
        if self.bin_ms not in {250, 500, 1_000}:
            raise ValueError("bin_ms must be one of 250, 500, or 1000")
        if not 0.5 <= self.train_fraction < 0.9:
            raise ValueError("train_fraction must be between 0.5 and 0.9")


APP_NAME = "Market Diffusion Lab"
APP_VERSION = "1.0.0"
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")

