import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df["기준시간"] = pd.to_datetime(df["기준시간"])
    logger.info(f"Loaded data: {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    df = df.copy()
    df["hour"] = df["기준시간"].dt.hour
    df["dayofweek"] = df["기준시간"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_daytime"] = ((df["기준시간"].dt.hour >= 7) & (df["기준시간"].dt.hour <= 22)).astype(int)

    # 마스크를 drop 전에 계산 (충전소ID·기준시간이 아직 존재할 때)
    mask_train = df["기준시간"] <= config.TRAIN_END
    mask_val = (df["기준시간"] > config.TRAIN_END) & (df["기준시간"] <= config.VAL_END)
    mask_test = df["기준시간"] > config.VAL_END

    # target encoding: train 기준으로만 계산 (leakage 방지)
    # 1차: 충전소별 평균
    station_mean = df.loc[mask_train].groupby("충전소ID")[target_col].mean()
    df["station_target_mean"] = df["충전소ID"].map(station_mean)
    df["station_target_mean"] = df["station_target_mean"].fillna(station_mean.mean())

    # 2차: 충전소 × 시간대 평균 (hour 파생 완료 이후)
    station_hour_mean = (
        df.loc[mask_train]
        .groupby(["충전소ID", "hour"])[target_col]
        .mean()
    )
    df["station_hour_mean"] = (
        df.set_index(["충전소ID", "hour"])
        .index.map(station_hour_mean)
    )
    df["station_hour_mean"] = df["station_hour_mean"].fillna(df["station_target_mean"])

    drop_cols = config.DROP_COLS + config.TARGET_COLS
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    for c in config.CAT_COLS:
        X[c] = X[c].astype("category")

    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]

    logger.info(f"[{target_col}] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test
