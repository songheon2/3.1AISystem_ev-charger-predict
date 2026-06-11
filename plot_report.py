import logging

import pandas as pd

import config
from src.dataset import load_data, split_data
from src.evaluate import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_trial_comparison,
)
from src.train import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    df = load_data(config.DATA_DIR)

    for target in config.TARGET_COLS:
        safe_name = target.replace("/", "_").replace("(", "_").replace(")", "_")
        logger.info(f"=== {target} 시각화 ===")

        _, _, X_test, _, _, y_test = split_data(
            df, target,
            train_end=config.FULL_CLF_TRAIN_END,
            val_end=config.FULL_CLF_VAL_END,
        )
        y_test_full = (y_test == 1.0).astype(int)

        clf = load_model(config.MODEL_DIR / f"lgbm_full_{safe_name}.pkl")
        threshold = float(
            pd.read_csv(config.METRICS_DIR / f"full_clf_metrics_{safe_name}.csv")
            ["best_threshold"].iloc[0]
        )
        proba = clf.predict(X_test)

        plot_pr_curve(
            y_test_full, proba, threshold,
            config.PLOT_DIR / f"pr_curve_{safe_name}.png",
        )
        plot_confusion_matrix(
            y_test_full, proba, threshold,
            config.PLOT_DIR / f"confusion_matrix_{safe_name}.png",
        )
        plot_class_distribution(
            y_test_full,
            config.PLOT_DIR / f"class_distribution_{safe_name}.png",
        )

    plot_trial_comparison(config.PLOT_DIR / "trial_comparison.png")


if __name__ == "__main__":
    main()
