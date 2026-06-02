import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


def load_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("ratio_dataset_*.csv"))
    if not files:
        raise FileNotFoundError(f"No ratio_dataset_*.csv files found in {data_dir}")
    dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
    logger.info(f"Loading {len(files)} file(s): {[f.name for f in files]}")
    df = pd.concat(dfs, ignore_index=True)
    df["기준시간"] = pd.to_datetime(df["기준시간"])
    logger.info(f"Loaded data: {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    train_end: str = config.TRAIN_END,
    val_end: str = config.VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    df = df.copy()
    df["hour"] = df["기준시간"].dt.hour
    df["dayofweek"] = df["기준시간"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_daytime"] = ((df["기준시간"].dt.hour >= 7) & (df["기준시간"].dt.hour <= 22)).astype(int)

    # lag 계산 전 정렬 (충전소ID별 시간 순서 보장)
    df = df.sort_values(["충전소ID", "기준시간"]).reset_index(drop=True)

    # 마스크를 drop 전에 계산 (충전소ID·기준시간이 아직 존재할 때)
    mask_train = df["기준시간"] <= train_end
    mask_val = (df["기준시간"] > train_end) & (df["기준시간"] <= val_end)
    mask_test = df["기준시간"] > val_end

    # Lag 피처 (split 전 전체 df에서 계산 — 경계 누수 없음, NaN은 LightGBM 자체 처리)
    df["lag_1h"] = df.groupby("충전소ID")[target_col].shift(1)
    df["lag_2h"] = df.groupby("충전소ID")[target_col].shift(2)
    df["lag_24h"] = df.groupby("충전소ID")[target_col].shift(24)

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
