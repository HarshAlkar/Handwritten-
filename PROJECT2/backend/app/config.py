"""Application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT2_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT2_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cors_origins: str = "http://localhost:5174,http://127.0.0.1:5174"
    artifacts_dir: Path = PROJECT2_ROOT / "artifacts"
    max_n_estimators: int = 500
    max_train_seconds: int = 120

    def models_dir(self) -> Path:
        path = self.artifacts_dir / "models"
        if not path.is_absolute():
            path = PROJECT2_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def metadata_dir(self) -> Path:
        path = self.artifacts_dir / "metadata"
        if not path.is_absolute():
            path = PROJECT2_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
