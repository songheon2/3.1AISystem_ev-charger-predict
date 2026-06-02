import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

import config

logger = logging.getLogger(__name__)


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.Booster:
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        config.LGBM_PARAMS,
        dtrain,
        num_boost_round=config.NUM_BOOST_ROUND,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    return model


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.Booster:
    params = {**config.LGBM_CLF_PARAMS}
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=config.NUM_BOOST_ROUND,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    logger.info(f"Classifier best iteration: {model.best_iteration}")
    return model


def train_full_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.Booster:
    params = {**config.LGBM_FULL_CLF_PARAMS}
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=config.NUM_BOOST_ROUND,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    logger.info(f"Full classifier best iteration: {model.best_iteration}")
    return model


def save_model(model: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: Path) -> lgb.Booster:
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model
