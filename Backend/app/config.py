"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: Path = PROJECT_ROOT / "ML" / "artifacts" / "digit_cnn.keras"
    history_path: Path = PROJECT_ROOT / "ML" / "artifacts" / "history.json"
    metrics_path: Path = PROJECT_ROOT / "ML" / "artifacts" / "metrics.json"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_bytes: int = 2_097_152

    def resolved_model_path(self) -> Path:
        path = self.model_path
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolved_history_path(self) -> Path:
        path = self.history_path
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolved_metrics_path(self) -> Path:
        path = self.metrics_path
        return path if path.is_absolute() else PROJECT_ROOT / path

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
