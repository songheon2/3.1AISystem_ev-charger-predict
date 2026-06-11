import logging
import random

import numpy as np

import config
from src.dataset import export_encodings, load_data, split_data
from src.evaluate import (
    compute_full_clf_metrics,
    find_threshold_for_recall,
    plot_feature_importance,
    save_metrics,
)
from src.train import save_model, train_full_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    df = load_data(config.DATA_DIR)

    for target in config.TARGET_COLS:
        logger.info(f"=== Target: {target} ===")
        safe_name = target.replace("/", "_").replace("(", "_").replace(")", "_")

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df, target,
            train_end=config.FULL_CLF_TRAIN_END,
            val_end=config.FULL_CLF_VAL_END,
        )

        y_train_full = (y_train == 1.0).astype(int)
        y_val_full = (y_val == 1.0).astype(int)
        y_test_full = (y_test == 1.0).astype(int)

        logger.info(
            f"[{target}] 만차 비율 — train: {y_train_full.mean():.4f}, "
            f"val: {y_val_full.mean():.4f}, test: {y_test_full.mean():.4f}"
        )

        # --- Negative Undersampling (train only) ---
        pos_idx = X_train[y_train_full == 1].index
        neg_idx = X_train[y_train_full == 0].sample(
            n=len(pos_idx) * config.FULL_CLF_NEG_RATIO,
            random_state=config.SEED,
        ).index
        sample_idx = pos_idx.union(neg_idx)
        X_train_s = X_train.loc[sample_idx]
        y_train_s = y_train_full.loc[sample_idx]
        logger.info(
            f"[{target}] Undersampled train: {len(X_train_s)} "
            f"(pos={len(pos_idx)}, neg={len(neg_idx)})"
        )

        clf = train_full_classifier(X_train_s, y_train_s, X_val, y_val_full)
        save_model(clf, config.MODEL_DIR / f"lgbm_full_{safe_name}.pkl")

        proba_val = clf.predict(X_val)
        best_threshold = find_threshold_for_recall(
            proba_val, y_val_full, config.FULL_CLF_TARGET_RECALL
        )

        proba_test = clf.predict(X_test)
        metrics = compute_full_clf_metrics(y_test_full, proba_test, best_threshold)
        save_metrics(metrics, config.METRICS_DIR / f"full_clf_metrics_{safe_name}.csv")
        plot_feature_importance(clf, config.PLOT_DIR / f"fi_full_{safe_name}.png")


    export_encodings(df, config.FULL_CLF_TRAIN_END)


if __name__ == "__main__":
    main()
