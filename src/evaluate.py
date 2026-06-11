import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rc("font", family="Malgun Gothic")
matplotlib.rc("axes", unicode_minus=False)
plt.rcParams["font.family"] = "Malgun Gothic"
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
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
        "recall":         float(recall_score(y_true, y_pred)),
        "f2":             float(fbeta_score(y_true, y_pred, beta=2)),
        "f1":             float(f1_score(y_true, y_pred)),
        "precision":      float(precision_score(y_true, y_pred)),
        "auc":            float(roc_auc_score(y_true, y_proba)),
        "pr_auc":         float(average_precision_score(y_true, y_proba)),
        "best_threshold": threshold,
    }


def plot_feature_importance(model: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    lgb.plot_importance(model, ax=ax, max_num_features=20, importance_type="gain")
    ax.set_title("Feature Importance (Gain)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance plot saved to {path}")


def plot_pr_curve(
    y_true: pd.Series,
    proba: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, proba)
    pr_auc = float(average_precision_score(y_true, proba))
    baseline = float(y_true.mean())

    # threshold에 가장 가까운 인덱스
    idx = int(np.argmin(np.abs(thresholds - threshold)))
    pt_recall = float(recall_vals[idx])
    pt_precision = float(precision_vals[idx])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, color="steelblue",
            label=f"PR curve (PR-AUC = {pr_auc:.3f})")
    ax.axhline(baseline, linestyle="--", color="gray",
               label=f"Baseline = {baseline:.3f}")
    ax.plot(pt_recall, pt_precision, "*", markersize=14, color="crimson",
            label=f"threshold = {threshold:.2f}")
    ax.annotate(
        f"(Recall={pt_recall:.3f}, Precision={pt_precision:.3f})",
        xy=(pt_recall, pt_precision),
        xytext=(pt_recall + 0.04, pt_precision + 0.04),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "crimson"},
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved to {path}")


def plot_confusion_matrix(
    y_true: pd.Series,
    proba: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["비만차", "만차"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {path}")


def plot_class_distribution(y_true: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = y_true.value_counts().sort_index()
    labels = ["비만차", "만차"]
    total = len(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, counts.values, color=["steelblue", "crimson"])
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}건 ({pct:.2f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("샘플 수")
    ax.set_title("클래스 분포 (Test set)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Class distribution saved to {path}")


def plot_trial_comparison(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    versions = [
        "v1.0 회귀\n(비율예측)",
        "v3.0\nscale_pos_weight=24",
        "v3.0\n언더샘플링 F2기준",
        "v3.0 최종\n(Recall≥0.85)",
    ]
    recalls    = [None, 0.01,  0.66,  0.862]
    precisions = [None, 0.30,  0.21,  0.113]

    x = np.arange(len(versions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    r_vals = [v if v is not None else 0.0 for v in recalls]
    p_vals = [v if v is not None else 0.0 for v in precisions]

    bars_r = ax.bar(x - width / 2, r_vals, width, label="Recall", color="steelblue")
    bars_p = ax.bar(x + width / 2, p_vals, width, label="Precision", color="coral")

    # v1.0은 회귀 모델 — 막대 숨김
    bars_r[0].set_visible(False)
    bars_p[0].set_visible(False)

    for i, (r, p) in enumerate(zip(recalls, precisions)):
        if r is not None:
            ax.text(x[i] - width / 2, r + 0.01, f"{r:.2f}",
                    ha="center", va="bottom", fontsize=9)
            ax.text(x[i] + width / 2, p + 0.01, f"{p:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("시행착오 비교 — Recall / Precision")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Trial comparison saved to {path}")
