from pathlib import Path
from typing import Optional, Any

import joblib

from .logging import logger

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def save_checkpoint(obj: Any, name: str):
    """Saves an object to the checkpoint directory."""
    path = CHECKPOINT_DIR / f"{name}.pkl"
    joblib.dump(obj, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(name: str) -> Optional[Any]:
    """Loads an object from the checkpoint directory."""
    path = CHECKPOINT_DIR / f"{name}.pkl"
    if path.exists():
        obj = joblib.load(path)
        logger.info(f"Checkpoint loaded: {path}")
        return obj
    else:
        logger.info(f"No checkpoint found for: {path}")
        return None
