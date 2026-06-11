import logging
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from src.train import load_model

logger = logging.getLogger(__name__)


def aggregate_stations(api_rows: list[dict]) -> pd.DataFrame:
    """API 충전기 목록 → csId별 현재 충전 비율"""
    df = pd.DataFrame(api_rows)
    records = []
    for cs_id, group in df.groupby("csId"):
        total = len(group)
        occupied = int((group["cpStat"] != 1).sum())
        records.append({"csId": str(cs_id), "ratio": occupied / total if total > 0 else 0.0})
    return pd.DataFrame(records)


def load_encodings() -> tuple[dict, dict]:
    """station_meta.pkl, station_encodings.pkl 로드"""
    meta = joblib.load(config.ENCODINGS_DIR / "station_meta.pkl")
    encodings = joblib.load(config.ENCODINGS_DIR / "station_encodings.pkl")
    return meta, encodings


def get_lag_features(csIds: list[str], current_dt: datetime) -> pd.DataFrame:
    """history CSV에서 lag_1h·lag_2h·lag_24h 조회"""
    target_dts = {
        "lag_1h": current_dt - timedelta(hours=1),
        "lag_2h": current_dt - timedelta(hours=2),
        "lag_24h": current_dt - timedelta(hours=24),
    }
    rows = {
        cs_id: {"csId": cs_id, "lag_1h": np.nan, "lag_2h": np.nan, "lag_24h": np.nan}
        for cs_id in csIds
    }

    if not config.HISTORY_FILE.exists():
        return pd.DataFrame(list(rows.values()))

    hist = pd.read_csv(config.HISTORY_FILE, dtype={"csId": str})
    hist["datetime"] = pd.to_datetime(hist["datetime"])

    for cs_id in csIds:
        cs_hist = hist[hist["csId"] == cs_id].set_index("datetime")["ratio"]
        if cs_hist.empty:
            continue
        for key, target_dt in target_dts.items():
            if target_dt in cs_hist.index:
                rows[cs_id][key] = float(cs_hist[target_dt])
            else:
                diffs = abs(cs_hist.index - target_dt)
                if diffs.min() <= timedelta(minutes=30):
                    rows[cs_id][key] = float(cs_hist.iloc[diffs.argmin()])

    return pd.DataFrame(list(rows.values()))


def build_features(
    station_df: pd.DataFrame,
    current_dt: datetime,
    meta: dict,
    encodings: dict,
    lag_df: pd.DataFrame,
) -> pd.DataFrame:
    """추론용 피처 행렬 조립 — 학습 피처와 동일한 컬럼 순서 보장"""
    hour = current_dt.hour
    dayofweek = current_dt.weekday()
    is_weekend = int(dayofweek >= 5)
    is_daytime = int(7 <= hour <= 22)

    lag_lookup = lag_df.set_index("csId")
    rows = []

    for _, srow in station_df.iterrows():
        cs_id = str(srow["csId"])
        m = meta.get(cs_id, {})
        enc = encodings.get(cs_id, {})
        lag = lag_lookup.loc[cs_id] if cs_id in lag_lookup.index else None

        station_mean = enc.get("station_target_mean", np.nan)
        hour_means = enc.get("hour_means", {})
        station_hour_mean = float(hour_means.get(hour, station_mean)) if hour_means else station_mean

        rows.append({
            "csId": cs_id,
            "유형": m.get("유형", np.nan),
            "시도": m.get("시도", np.nan),
            "시군구": m.get("시군구", np.nan),
            "hour": hour,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
            "is_daytime": is_daytime,
            "station_target_mean": station_mean,
            "station_hour_mean": station_hour_mean,
            "lag_1h": lag["lag_1h"] if lag is not None else np.nan,
            "lag_2h": lag["lag_2h"] if lag is not None else np.nan,
            "lag_24h": lag["lag_24h"] if lag is not None else np.nan,
        })

    return pd.DataFrame(rows)


def update_history(station_df: pd.DataFrame, current_dt: datetime) -> None:
    """현재 ratio를 history CSV에 append"""
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    new_rows = pd.DataFrame({
        "csId": station_df["csId"].astype(str),
        "datetime": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "ratio": station_df["ratio"].values,
    })
    write_header = not config.HISTORY_FILE.exists()
    new_rows.to_csv(config.HISTORY_FILE, mode="a", index=False, header=write_header)
    logger.info(f"History updated: {len(new_rows)} rows at {current_dt}")


def predict_realtime(api_rows: list[dict]) -> pd.DataFrame:
    """전체 파이프라인 실행 — 외부에서 호출하는 단일 진입점

    Returns: DataFrame[csId, proba_1h, full_1h, proba_2h, full_2h]
    """
    # [1] aggregate
    station_df = aggregate_stations(api_rows)

    # statUpdateDatetime: yyyyMMddHHmmss → datetime
    dt_str = str(api_rows[0]["statUpdateDatetime"])
    current_dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")

    # [2] load encodings
    meta, encodings = load_encodings()

    # [3] lag features
    lag_df = get_lag_features(station_df["csId"].tolist(), current_dt)

    # [4] build features
    feat_df = build_features(station_df, current_dt, meta, encodings, lag_df)

    # [5] predict
    result = feat_df[["csId"]].copy()

    for target in config.TARGET_COLS:
        safe_name = target.replace("/", "_").replace("(", "_").replace(")", "_")
        clf = load_model(config.MODEL_DIR / f"lgbm_full_{safe_name}.pkl")
        threshold = float(
            pd.read_csv(config.METRICS_DIR / f"full_clf_metrics_{safe_name}.csv")
            ["best_threshold"].iloc[0]
        )

        # 학습 시 피처 순서에 맞춰 정렬, 없는 컬럼은 NaN
        feature_names = clf.feature_name()
        X = feat_df.reindex(columns=feature_names)
        for c in config.CAT_COLS:
            if c in X.columns:
                X[c] = X[c].astype("category")

        proba = clf.predict(X)
        full = (proba >= threshold).astype(int)

        suffix = "1h" if "1시간" in target else "2h"
        result[f"proba_{suffix}"] = proba
        result[f"full_{suffix}"] = full

    # [6] update history
    update_history(station_df, current_dt)

    logger.info(f"predict_realtime: {len(result)} stations at {current_dt}")
    return result


def predict_realtime_from_csv(csv_path: Path) -> pd.DataFrame:
    """CSV 파일 → predict_realtime() 호출 — 외부 최종 진입점

    CSV 컬럼: csId, cpStat, statUpdateDatetime (yyyyMMddHHmmss)
    실데이터 호환: statUpdatedatetime NaN 시 snapshot_dt (yyyyMMdd_HHmmss) 폴백
    Returns: DataFrame[csId, proba_1h, full_1h, proba_2h, full_2h]
    """
    df = pd.read_csv(csv_path, dtype={"csId": str})

    # statUpdateDatetime 정규화
    if "statUpdateDatetime" not in df.columns:
        if "statUpdatedatetime" in df.columns and df["statUpdatedatetime"].notna().any():
            df["statUpdateDatetime"] = df["statUpdatedatetime"].astype(str)
        elif "snapshot_dt" in df.columns:
            # yyyyMMdd_HHmmss → yyyyMMddHHmmss
            df["statUpdateDatetime"] = df["snapshot_dt"].astype(str).str.replace("_", "", regex=False)
        else:
            raise ValueError("datetime 컬럼을 찾을 수 없습니다 (statUpdateDatetime / snapshot_dt)")

    api_rows = df[["csId", "cpStat", "statUpdateDatetime"]].to_dict(orient="records")
    return predict_realtime(api_rows)
