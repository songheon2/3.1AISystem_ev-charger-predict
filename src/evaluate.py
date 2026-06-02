import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", family="Malgun Gothic")
matplotlib.rc("axes", unicode_minus=False)
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
)

import config

logger = logging.getLogger(__name__)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mask = y_true != 0
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_nonzero": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
    }
    return metrics


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)
    logger.info(f"Metrics saved to {path}")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")


def compute_clf_metrics(y_true: pd.Series, y_pred_proba: np.ndarray) -> dict[str, float]:
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_pred_proba)),
        "f1": float(f1_score(y_true, y_pred_binary)),
        "precision": float(precision_score(y_true, y_pred_binary)),
        "recall": float(recall_score(y_true, y_pred_binary)),
    }


def find_threshold_for_recall(
    y_proba: np.ndarray,
    y_true: pd.Series,
    target_recall: float,
) -> float:
    thresholds = np.arange(0.01, 0.50, 0.01)
    valid = [t for t in thresholds if recall_score(y_true, y_proba >= t) >= target_recall]
    return float(max(valid)) if valid else float(thresholds[0])


def compute_full_clf_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "recall": float(recall_score(y_true, y_pred)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "best_threshold": threshold,
    }


def plot_feature_importance(model: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    lgb.plot_importance(model, ax=ax, max_num_features=20, importance_type="gain")
    ax.set_title("Feature Importance (Gain)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Feature importance plot saved to {path}")
