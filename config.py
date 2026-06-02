from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "outputs" / "models"
PLOT_DIR = ROOT_DIR / "outputs" / "plots"
METRICS_DIR = ROOT_DIR / "outputs" / "metrics"

# --- Data ---
SEED = 42

TARGET_COLS: list[str] = ["1시간뒤_정답(y)", "2시간뒤_정답(y)"]
CAT_COLS: list[str] = ["유형", "시도", "시군구"]
DROP_COLS: list[str] = ["충전소ID", "기준시간"]

# 시간 기반 split 기준 (랜덤 split 사용 금지 — 미래 누수 방지)
TRAIN_END = "2025-10-31 23:00:00"
VAL_END = "2025-11-30 23:00:00"
# test: 2025-12-01 ~

# --- LightGBM Hyperparameters ---
LGBM_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "verbose": -1,
    "seed": SEED,
}

NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 100

# --- Stage 1: Classifier ---
LGBM_CLF_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": SEED,
}
CLF_THRESHOLD: float = 0.08

# --- v3.0: Full Classifier (만차 여부 이진 분류) ---
FULL_CLF_DATA_FILE: str = "ratio_dataset_202501.csv"
FULL_CLF_TRAIN_END: str = "2025-01-24 23:00:00"
FULL_CLF_VAL_END: str = "2025-01-27 23:00:00"
FULL_CLF_NEG_RATIO: int = 5
FULL_CLF_TARGET_RECALL: float = 0.85

LGBM_FULL_CLF_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": SEED,
}
