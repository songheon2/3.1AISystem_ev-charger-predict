import logging
import random

import numpy as np

import config
from src.dataset import load_data, split_data
from src.evaluate import compute_clf_metrics, compute_metrics, plot_feature_importance, save_metrics
from src.train import save_model, train, train_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    data_path = config.DATA_DIR / config.DATA_FILE
    df = load_data(data_path)

    for target in config.TARGET_COLS:
        logger.info(f"=== Target: {target} ===")
        safe_name = target.replace("/", "_").replace("(", "_").replace(")", "_")

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target)

        # --- Stage 1: 분류기 ---
        y_train_binary = (y_train != 0).astype(int)
        y_val_binary = (y_val != 0).astype(int)
        y_test_binary = (y_test != 0).astype(int)

        clf = train_classifier(X_train, y_train_binary, X_val, y_val_binary)
        save_model(clf, config.MODEL_DIR / f"lgbm_clf_{safe_name}.pkl")

        # val set 기준 F1 최대화 threshold 탐색
        clf_proba_val = clf.predict(X_val)
        thresholds = np.arange(0.05, 0.30, 0.01)
        from sklearn.metrics import f1_score as _f1
        f1_scores = [_f1(y_val_binary, clf_proba_val >= t) for t in thresholds]
        best_t = float(thresholds[np.argmax(f1_scores)])
        logger.info(f"[{target}] Best threshold (val F1): {best_t:.2f}  (config: {config.CLF_THRESHOLD})")

        clf_proba = clf.predict(X_test)
        clf_metrics = compute_clf_metrics(y_test_binary, clf_proba)
        save_metrics(clf_metrics, config.METRICS_DIR / f"clf_metrics_{safe_name}.csv")
        plot_feature_importance(clf, config.PLOT_DIR / f"fi_clf_{safe_name}.png")

        # --- Stage 2: 회귀기 (non-zero 행만) ---
        mask_nz_train = y_train != 0
        mask_nz_val = y_val != 0

        reg = train(
            X_train[mask_nz_train], y_train[mask_nz_train],
            X_val[mask_nz_val], y_val[mask_nz_val],
        )
        save_model(reg, config.MODEL_DIR / f"lgbm_reg_{safe_name}.pkl")

        # --- 최종 예측 결합 ---
        ratio_pred = reg.predict(X_test)
        final_pred = np.where(clf_proba >= config.CLF_THRESHOLD, ratio_pred, 0.0)

        all_metrics = compute_metrics(y_test, final_pred)
        final_metrics = {"rmse_nonzero": all_metrics["rmse_nonzero"]}
        save_metrics(final_metrics, config.METRICS_DIR / f"metrics_{safe_name}.csv")
        plot_feature_importance(reg, config.PLOT_DIR / f"fi_reg_{safe_name}.png")


if __name__ == "__main__":
    main()
