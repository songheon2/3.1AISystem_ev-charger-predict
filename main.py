import logging
import random

import numpy as np

import config
from src.dataset import load_data, split_data
from src.evaluate import compute_metrics, plot_feature_importance, save_metrics
from src.train import save_model, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    # 1. Load data
    data_path = config.DATA_DIR / config.DATA_FILE
    df = load_data(data_path)

    # 2. 두 타겟 각각 학습·평가
    for target in config.TARGET_COLS:
        logger.info(f"=== Target: {target} ===")

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target)

        model = train(X_train, y_train, X_val, y_val)

        safe_name = target.replace("/", "_").replace("(", "_").replace(")", "_")
        save_model(model, config.MODEL_DIR / f"lgbm_{safe_name}.pkl")

        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        save_metrics(metrics, config.METRICS_DIR / f"metrics_{safe_name}.csv")

        plot_feature_importance(model, config.PLOT_DIR / f"fi_{safe_name}.png")


if __name__ == "__main__":
    main()
