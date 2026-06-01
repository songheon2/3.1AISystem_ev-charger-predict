import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.train import load_model

logger = logging.getLogger(__name__)


def predict(model_path: Path, X: pd.DataFrame) -> np.ndarray:
    model = load_model(model_path)
    preds = model.predict(X)
    logger.info(f"Prediction done: {preds.shape}")
    return preds
