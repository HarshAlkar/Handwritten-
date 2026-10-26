"""Training configuration with reproducible defaults."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "ML" / "artifacts"


@dataclass
class TrainConfig:
    dataset_name: str = "ylecun/mnist"
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    random_seed: int = 42
    validation_split: float = 0.1
    early_stopping_patience: int = 3
    model_path: Path = ARTIFACTS_DIR / "digit_cnn.keras"
    history_path: Path = ARTIFACTS_DIR / "history.json"
    checkpoint_path: Path = ARTIFACTS_DIR / "checkpoints" / "best.keras"

    @classmethod
    def from_env(cls) -> "TrainConfig":
        cfg = cls()
        if os.getenv("EPOCHS"):
            cfg.epochs = int(os.getenv("EPOCHS", cfg.epochs))
        if os.getenv("BATCH_SIZE"):
            cfg.batch_size = int(os.getenv("BATCH_SIZE", cfg.batch_size))
        if os.getenv("LEARNING_RATE"):
            cfg.learning_rate = float(os.getenv("LEARNING_RATE", cfg.learning_rate))
        if os.getenv("RANDOM_SEED"):
            cfg.random_seed = int(os.getenv("RANDOM_SEED", cfg.random_seed))
        if os.getenv("VALIDATION_SPLIT"):
            cfg.validation_split = float(os.getenv("VALIDATION_SPLIT", cfg.validation_split))
        if os.getenv("MODEL_PATH"):
            cfg.model_path = Path(os.getenv("MODEL_PATH", str(cfg.model_path)))
            if not cfg.model_path.is_absolute():
                cfg.model_path = PROJECT_ROOT / cfg.model_path
        if os.getenv("HISTORY_PATH"):
            cfg.history_path = Path(os.getenv("HISTORY_PATH", str(cfg.history_path)))
            if not cfg.history_path.is_absolute():
                cfg.history_path = PROJECT_ROOT / cfg.history_path
        return cfg

    def to_dict(self) -> dict:
        data = asdict(self)
        data["model_path"] = str(self.model_path)
        data["history_path"] = str(self.history_path)
        data["checkpoint_path"] = str(self.checkpoint_path)
        return data
