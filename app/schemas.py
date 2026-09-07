"""HTTP request models."""

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    seed: int = Field(default=42, ge=0, le=999_999)
    duration_seconds: int = Field(default=1_200, ge=300, le=7_200)
    bin_ms: int = Field(default=1_000)

