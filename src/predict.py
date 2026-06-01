import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.train import load_model

logger = logging.getLogger(__name__)


def predict(model_path: Path, X: pd.DataFrame) -> np.ndarray:
    model = load_model(model_path)
    preds = model.predict(X)
    logger.info(f"Prediction done: {preds.shape}")
    return preds


def predict_twostage(
    clf_path: Path,
    reg_path: Path,
    X: pd.DataFrame,
    threshold: float = config.CLF_THRESHOLD,
) -> np.ndarray:
    clf = load_model(clf_path)
    reg = load_model(reg_path)

    proba = clf.predict(X)
    ratio = reg.predict(X)
    final = np.where(proba >= threshold, ratio, 0.0)

    logger.info(f"Two-stage prediction done: {final.shape}, non-zero ratio: {(final != 0).mean():.3f}")
    return final
