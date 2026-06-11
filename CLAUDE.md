# EV 충전기 예측 시스템 — 설계 명세

## 프로젝트 개요

- **데이터**: `ratio_dataset_202501.csv` ~ `ratio_dataset_202512.csv` (2025-01-01 ~ 2025-12-31, 월별 파일 12개)
- **충전소 수**: 2,069개
- **목표**: 1시간 뒤 / 2시간 뒤 만차 여부 예측 (이진 분류)
- **모델**: LightGBM (v3.0 만차 분류기)

---

## 프로젝트 구조

```
ev-charger-predict/
├── src/
│   ├── __init__.py
│   ├── dataset.py      # 데이터 로드, 피처 엔지니어링, 시간 기반 split
│   ├── train.py        # LightGBM 학습, 모델 저장/로드
│   ├── evaluate.py     # 메트릭 계산, 플롯 저장
│   ├── predict.py      # 추론 (배치)
│   └── infer.py        # 실시간 추론 파이프라인 (API 데이터 → 만차 예측)
├── outputs/
│   ├── models/         # lgbm_*.pkl
│   ├── plots/          # fi_*.png
│   ├── metrics/        # metrics_*.csv
│   ├── encodings/      # station_meta.pkl, station_encodings.pkl
│   └── history/        # station_history.csv (lag 이력)
├── data/
│   ├── processed/      # ratio_dataset_202501.csv ~ ratio_dataset_202512.csv (월별 12개)
│   └── realtime/       # 실시간 API CSV 입력 파일
├── config.py           # 모든 상수·하이퍼파라미터
├── main.py             # 학습 파이프라인
├── CLAUDE.md           # 이 파일 (single source of truth)
└── CLAUDE.local.md     # 코딩 담당 전용 로컬 기록 (git 제외)
```

---

## 데이터 명세

| 항목 | 내용 |
|------|------|
| 인코딩 | UTF-8 BOM → `encoding='utf-8-sig'` |
| 타겟 | `1시간뒤_정답(y)`, `2시간뒤_정답(y)` |
| 범주형 | `유형`(5), `시도`(17), `시군구`(216) → category dtype |
| 제외 컬럼 | `충전소ID`, `기준시간` (DROP_COLS) |
| 결측 | 기온, 강수량 각 32,857건 — LightGBM 자체 처리 |
| 데이터 품질 | 4월 데이터에 `ME#S` 접두 이상 ID 750개 확인 → 로드 시 제거 |
| 타겟 분포 | 0.0이 약 91.9% (극도 불균형) |

---

## Split 전략

- **방식**: 시간 기반 split (랜덤 split 금지 — 미래 누수 방지)
- Train: `~ 2025-10-31 23:00` (10개월)
- Val: `2025-11-01 00:00 ~ 2025-11-30 23:00` (1개월)
- Test: `2025-12-01 00:00 ~` (1개월)

---

## 피처 엔지니어링

- `기준시간` → `hour`, `dayofweek`, `is_weekend`, `is_daytime` 파생 후 원본 drop
- `CAT_COLS` → `category` dtype 변환 (LightGBM 자동 인식)
- target encoding: `station_target_mean`, `station_hour_mean` (train 기간 기준, leakage 없음)

### Lag 피처 (v2.1 추가)

| 피처 | 의미 | 계산 |
|------|------|------|
| `lag_1h` | 1시간 전 충전 비율 | `groupby('충전소ID')[target_col].shift(1)` |
| `lag_2h` | 2시간 전 충전 비율 | `groupby('충전소ID')[target_col].shift(2)` |
| `lag_24h` | 24시간 전 충전 비율 (전일 동시간대) | `groupby('충전소ID')[target_col].shift(24)` |

**구현 규칙**
- 전체 df를 `(충전소ID, 기준시간)` 기준으로 정렬한 뒤 lag 계산 (split 전)
- split 경계를 자연스럽게 넘어 val/test 행도 train 과거값을 lag로 참조 — 정상 동작
- 결측(NaN) 그대로 유지 — LightGBM 자체 처리
- `target_col`에서 shift하므로 target마다 독립 계산 (main.py 루프 내 split_data 호출로 자동 처리)

---

## 모듈별 역할 및 인터페이스

### `config.py`
- 모든 상수·하이퍼파라미터 중앙 관리

### `src/dataset.py`
- `load_data(data_dir: Path) -> DataFrame`: `ratio_dataset_*.csv` 전체 glob → 시간순 정렬 후 concat, `기준시간` datetime 변환, 비정상 충전소ID(`ME#S` 접두) 제거
- `split_data(df, target_col) -> (X_train, X_val, X_test, y_train, y_val, y_test)`: 피처 엔지니어링 + lag 피처 + 시간 기반 split

### `src/train.py`
- `train(X_train, y_train, X_val, y_val) -> Booster`: 샘플 가중치 포함 학습
- `save_model / load_model`

### `src/evaluate.py`
- `compute_metrics(y_true, y_pred) -> dict`: RMSE, MAE, R²
- `save_metrics`, `plot_feature_importance`
- 한글 폰트: 맑은 고딕 적용

### `main.py`
- `TARGET_COLS` 루프로 두 타겟 순차 학습·평가

---

## 역할 정의

### [설계 담당] — 창 1
- **목적**: 아키텍처 설계 유지, 모듈 간 인터페이스 정의, CLAUDE.md를 single source of truth로 관리
- **권한**: CLAUDE.md 수정, 설계 변경 결정, 코딩 담당에게 지시 발행
- **금지**: 소스 코드 직접 작성·수정 (CLAUDE.md 제외)
- **의사결정 책임**:
  - 피처 추가/제거 여부
  - 모델 구조 변경 (단일 → 2단계 등)
  - 하이퍼파라미터 조정 방향
  - 평가 지표 기준 설정

### [코딩 담당] — 창 2
- **목적**: CLAUDE.md 및 설계 담당 지시 기반으로 실제 코드 구현
- **권한**: `src/`, `config.py`, `main.py`, `requirements.txt`, `CLAUDE.local.md` 수정
- **금지**: 설계 담당 승인 없이 독자적 피처·모델 구조 변경
- **의무**:
  - 하이퍼파라미터는 `config.py`에서만 로드, 하드코딩 금지
  - 모든 함수에 타입 힌트 필수
  - `print` 대신 `logging` 사용
  - 학습 결과(RMSE, rmse_nonzero, 조기종료 라운드) 반드시 설계 담당에게 보고
  - 설계 의도와 충돌하는 구현 이슈 발견 시 즉시 보고 후 대기

### [리뷰 담당] — 창 3
- **목적**: 구현 코드가 설계 의도와 일치하는지 검토
- **권한**: 개선 제안 (창 1 승인 후 창 2에 전달)
- **금지**: 코드 직접 수정, 창 1 승인 없이 창 2에 직접 변경 지시
- **보고 포맷**: `[PASS]` / `[WARN]` / `[FAIL]`

### 의사결정 흐름

```
창 2 (코딩) ──결과 보고──▶ 창 1 (설계) ──판단──▶ 창 2에 지시
창 3 (리뷰) ──개선 제안──▶ 창 1 (설계) ──승인/기각──▶ 창 2에 지시 or 유지
```

---

## 코딩 규칙

| 규칙 | 내용 |
|------|------|
| 상수·하이퍼파라미터 | `config.py`에서만 관리, 하드코딩 금지 |
| 타입 힌트 | 모든 함수 시그니처 필수 |
| 로깅 | `print` 대신 `logging` 모듈 |
| 시드 | `SEED = 42` |
| 스타일 | 순수 스크립트, Jupyter 셀 방식 금지 |

---

## v2.0 설계 — 2단계 모델

### 배경

v1.0 단일 회귀 모델은 타겟 91.9%가 0인 구조 때문에 비영 구간 예측에 편향(rmse_nonzero 0.62, 단순 평균 baseline ~0.38보다 낮음). 구조적 해결책으로 2단계 분리.

### 모델 구조

```
입력 피처 (18개)
      │
      ▼
[Stage 1] LightGBM Classifier  →  P(non-zero)
      │
      ├── P < CLF_THRESHOLD  →  예측값 = 0.0
      │
      └── P ≥ CLF_THRESHOLD  →  [Stage 2] LightGBM Regressor  →  예측 비율
```

**모델 파일 (타겟 2개 × 스테이지 2개 = 4개)**
- `lgbm_clf_1시간뒤_정답_y_.pkl` — Stage 1 분류기
- `lgbm_reg_1시간뒤_정답_y_.pkl` — Stage 2 회귀기
- `lgbm_clf_2시간뒤_정답_y_.pkl`
- `lgbm_reg_2시간뒤_정답_y_.pkl`

### v2.0 공식 평가 지표

| 단계 | 지표 | 의미 |
|------|------|------|
| Stage 1 | AUC, F1, Precision, Recall | 분류기 성능 |
| Stage 2 | rmse_nonzero | 회귀기 핵심 성능 |

> overall RMSE, R²는 2단계 모델에서 의미 없으므로 저장하지 않음

### Stage 1 — 분류기

| 항목 | 내용 |
|------|------|
| 학습 데이터 | 전체 train 행 |
| 타겟 | `(y != 0).astype(int)` |
| objective | `binary` |
| metric | `binary_logloss` |
| 불균형 처리 | 없음 (scale_pos_weight 제거 — AUC 0.82로 순위 구분 충분, 가중치는 학습 불안정 유발) |
| 평가 지표 | AUC, F1, Precision, Recall |

### Stage 2 — 회귀기

| 항목 | 내용 |
|------|------|
| 학습 데이터 | **train 중 non-zero 행만** |
| val 데이터 | **val 중 non-zero 행만** |
| 타겟 | 실제 충전 비율 |
| objective | `regression` (v1.0과 동일) |
| 평가 지표 | rmse_nonzero |

### config.py 추가 항목

```python
# Stage 1 — Classifier
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
CLF_THRESHOLD: float = 0.08   # non-zero 비율(~8.1%) 기반 설정
```

### 모듈별 변경 명세

| 파일 | 변경 내용 |
|------|-----------|
| `config.py` | `LGBM_CLF_PARAMS`, `CLF_THRESHOLD` 추가 |
| `src/train.py` | `train_classifier()` 함수 추가 |
| `src/evaluate.py` | `compute_clf_metrics()` 함수 추가 (AUC, F1, Precision, Recall) |
| `src/predict.py` | `predict_twostage()` 함수 추가 |
| `main.py` | 2단계 파이프라인으로 전면 교체, final_metrics는 rmse_nonzero만 저장 |

### main.py 파이프라인 흐름

```python
for target in TARGET_COLS:

    # Stage 1
    y_binary = (y != 0).astype(int)
    clf = train_classifier(X_train, y_train_binary, X_val, y_val_binary)
    save_model(clf, MODEL_DIR / f"lgbm_clf_{safe_name}.pkl")
    evaluate & save clf_metrics  # AUC, F1, Precision, Recall

    # Stage 2
    mask_nz_train = y_train != 0
    mask_nz_val   = y_val   != 0
    reg = train(X_train[mask_nz_train], y_train[mask_nz_train],
                X_val[mask_nz_val],   y_val[mask_nz_val])
    save_model(reg, MODEL_DIR / f"lgbm_reg_{safe_name}.pkl")

    # 최종 예측 (결합) — rmse_nonzero만 저장
    proba = clf.predict(X_test)
    ratio = reg.predict(X_test)
    final_pred = np.where(proba >= CLF_THRESHOLD, ratio, 0.0)
    save_metrics({"rmse_nonzero": ...}, ...)
```

---

## v3.0 설계 — 만차 분류 모델

### 목적 전환

실제 사용 목적이 "1시간/2시간 뒤 충전소가 꽉 차는지 여부"이므로, 비율 회귀 대신 **만차 여부 이진 분류**로 전환.

### 모델 구조

```
입력 피처 (lag 포함)
      │
      ▼
[LightGBM Classifier]  →  P(만차)
      │
      ├── P < FULL_CLF_THRESHOLD  →  비만차
      └── P ≥ FULL_CLF_THRESHOLD  →  만차
```

### 데이터 설정

| 항목 | 내용 |
|------|------|
| 학습 데이터 | **12개월 (ratio_dataset_202501.csv ~ 202512.csv)** |
| 로드 방식 | `load_data(DATA_DIR)` — glob 방식 |
| 타겟 | `(y == 1.0).astype(int)` — 만차 여부 |
| 클래스 분포 | 만차 3.92% / 비만차 96.08% |

### Split (12개월 기준)

| 구간 | 기간 |
|------|------|
| Train | ~ 2025-10-31 23:00 (10개월) |
| Val | 2025-11-01 00:00 ~ 2025-11-30 23:00 (1개월) |
| Test | 2025-12-01 00:00 ~ (1개월) |

### 불균형 처리

**Negative Undersampling (train only)**
- train set에서 negative를 `FULL_CLF_NEG_RATIO = 5` 비율로 random sampling
- positive : negative = 1 : 5 (원본 1:25 → 완화)
- val / test는 원본 분포 유지 (실제 운용 환경 반영)
- `random_state = config.SEED` 고정

### threshold 탐색

val set 기준 **Recall ≥ `FULL_CLF_TARGET_RECALL`(0.85)를 만족하는 최대 threshold** 탐색.
최대 threshold를 택해 Precision 손실 최소화.

```python
thresholds = np.arange(0.01, 0.50, 0.01)
valid = [t for t in thresholds if recall_score(y_val, proba >= t) >= config.FULL_CLF_TARGET_RECALL]
best_threshold = max(valid) if valid else thresholds[0]
```

목표 Recall을 만족하는 후보가 없으면 가장 낮은 threshold(0.01) 사용.

### 평가 지표 (우선순위 순)

| 순위 | 지표 | 의미 |
|------|------|------|
| 1 | Recall | 실제 만차를 얼마나 잡는가 (핵심) |
| 2 | F2 | Recall 2배 가중 조화평균 |
| 3 | F1 | 균형 지표 |
| 4 | Precision | 만차 예측 중 실제 만차 비율 |
| 5 | AUC | ROC 곡선 기반 분리 능력 |
| 6 | PR-AUC | Precision-Recall 곡선 아래 면적 — 불균형 데이터에서 AUC보다 정직한 성능 지표 |

### config.py 추가 항목

```python
FULL_CLF_TRAIN_END: str = "2025-10-31 23:00:00"
FULL_CLF_VAL_END: str   = "2025-11-30 23:00:00"
FULL_CLF_NEG_RATIO: int = 5        # train negative : positive 비율
FULL_CLF_TARGET_RECALL: float = 0.85  # threshold 탐색 목표 Recall

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
```

### 모듈별 변경 명세

| 파일 | 변경 내용 |
|------|-----------|
| `config.py` | `FULL_CLF_NEG_RATIO`, `FULL_CLF_TARGET_RECALL` 추가 |
| `src/evaluate.py` | `find_best_threshold_f2` → `find_threshold_for_recall()` 교체 |
| `main.py` | undersampling 블록 추가, threshold 탐색 함수 교체 |

### main.py 파이프라인 흐름

```python
df = load_data(config.DATA_DIR)  # 12개월 전체 glob 로드

for target in TARGET_COLS:
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, target,
        train_end=config.FULL_CLF_TRAIN_END,
        val_end=config.FULL_CLF_VAL_END,
    )

    y_train_full = (y_train == 1.0).astype(int)
    y_val_full   = (y_val   == 1.0).astype(int)
    y_test_full  = (y_test  == 1.0).astype(int)

    # --- Negative Undersampling (train only) ---
    pos_idx = X_train[y_train_full == 1].index
    neg_idx = X_train[y_train_full == 0].sample(
        n=len(pos_idx) * config.FULL_CLF_NEG_RATIO,
        random_state=config.SEED,
    ).index
    sample_idx = pos_idx.union(neg_idx)
    X_train_s = X_train.loc[sample_idx]
    y_train_s = y_train_full.loc[sample_idx]

    clf = train_full_classifier(X_train_s, y_train_s, X_val, y_val_full)
    save_model(clf, MODEL_DIR / f"lgbm_full_{safe_name}.pkl")

    # val Recall >= TARGET_RECALL 만족하는 최대 threshold 탐색
    proba_val = clf.predict(X_val)
    best_threshold = find_threshold_for_recall(proba_val, y_val_full, config.FULL_CLF_TARGET_RECALL)

    # test 평가
    proba_test = clf.predict(X_test)
    metrics = compute_full_clf_metrics(y_test_full, proba_test, best_threshold)
    save_metrics(metrics, METRICS_DIR / f"full_clf_metrics_{safe_name}.csv")
    plot_feature_importance(clf, PLOT_DIR / f"fi_full_{safe_name}.png")
```

### 모델 파일

- `lgbm_full_1시간뒤_정답_y_.pkl`
- `lgbm_full_2시간뒤_정답_y_.pkl`

---

## 실시간 추론 모듈 설계 (src/infer.py)

### API 데이터 명세

| 컬럼 | 설명 | 비고 |
|------|------|------|
| csId | 충전소ID | 그룹 키 |
| cpStat | 충전기 상태 | 1=사용가능, 나머지(2~7)=점유 |
| statUpdateDatetime | 상태 갱신 시각 | `yyyyMMddHHmmss` 형식 |

**점유 판정**: `cpStat != 1` → 점유  
**충전소 비율**: `(cpStat != 1).sum() / 전체_충전기_수`

---

### 추론 파이프라인

```
실시간 API 응답 (충전기 목록)
        │
        ▼
[1] aggregate_stations(api_rows)
    csId 기준 그룹화
    ratio = (cpStat != 1 수) / (전체 충전기 수)
        │
        ▼
[2] build_features(station_df, current_dt)
    시간 파생: hour, dayofweek, is_weekend, is_daytime
    메타 조회: 유형·시도·시군구 ← station_meta.pkl
    인코딩 조회: station_target_mean·station_hour_mean ← station_encodings.pkl
    lag 조회: lag_1h·lag_2h·lag_24h ← station_history.csv
        │
        ▼
[3] predict(feature_df)
    lgbm_full_1시간뒤.pkl → P(만차_1h)
    lgbm_full_2시간뒤.pkl → P(만차_2h)
    threshold 적용 → 만차 여부 (bool)
        │
        ▼
[4] update_history(station_df, current_dt)
    현재 ratio를 station_history.csv에 append
        │
        ▼
결과: csId별 {proba_1h, full_1h, proba_2h, full_2h}
```

---

### 저장 파일 명세

#### `outputs/encodings/station_meta.pkl`
```python
# dict: csId(str) → {"유형": ..., "시도": ..., "시군구": ...}
{
    "663": {"유형": "업무·관공서형", "시도": "서울특별시", "시군구": "관악구"},
    ...
}
```
- **생성 시점**: main.py 학습 완료 후
- **생성 방법**: 학습 데이터에서 csId별 첫 행 기준 추출

#### `outputs/encodings/station_encodings.pkl`
```python
# dict: csId(str) → {"station_target_mean": float, "hour_means": {hour(int): float}}
{
    "663": {"station_target_mean": 0.12, "hour_means": {0: 0.05, 1: 0.03, ..., 23: 0.08}},
    ...
}
```
- **생성 시점**: main.py 학습 완료 후
- **생성 방법**: train 기간 기준으로 계산된 값 추출

#### `outputs/history/station_history.csv`
```
csId,datetime,ratio
663,2025-12-01 00:00:00,0.5
663,2025-12-01 01:00:00,0.75
...
```
- **업데이트**: 매 추론 호출 시 현재 ratio append
- **조회**: lag_1h → current_dt - 1h, lag_2h → -2h, lag_24h → -24h 가장 가까운 행
- **NaN 처리**: 이력 없으면 NaN → LightGBM 자체 처리

---

### `src/infer.py` 함수 인터페이스

```python
def load_realtime_csv(csv_path: Path) -> list[dict]:
    """data/realtime/ CSV → list[dict] 변환"""
    # statUpdateDatetime: str → datetime 파싱 포함

def aggregate_stations(api_rows: list[dict]) -> pd.DataFrame:
    """API 충전기 목록 → csId별 현재 충전 비율"""

def load_encodings() -> tuple[dict, dict]:
    """station_meta.pkl, station_encodings.pkl 로드"""

def get_lag_features(csIds: list[str], current_dt: datetime) -> pd.DataFrame:
    """history CSV에서 lag_1h·lag_2h·lag_24h 조회"""

def build_features(
    station_df: pd.DataFrame,
    current_dt: datetime,
    meta: dict,
    encodings: dict,
    lag_df: pd.DataFrame,
) -> pd.DataFrame:
    """추론용 피처 행렬 조립 — 학습 피처와 동일한 컬럼 순서 보장"""

def update_history(station_df: pd.DataFrame, current_dt: datetime) -> None:
    """현재 ratio를 history CSV에 append"""

def predict_realtime(api_rows: list[dict]) -> pd.DataFrame:
    """전체 파이프라인 실행 — 외부에서 호출하는 단일 진입점"""
    # Returns: DataFrame[csId, proba_1h, full_1h, proba_2h, full_2h]

def predict_realtime_from_csv(csv_path: Path) -> pd.DataFrame:
    """CSV 파일 경로를 받아 predict_realtime() 실행하는 편의 함수"""
```

---

### main.py 추가 — 인코딩 저장

학습 완료 후 아래 단계 추가:

```python
from src.dataset import export_encodings
export_encodings(df, config.FULL_CLF_TRAIN_END)
# → outputs/encodings/station_meta.pkl
# → outputs/encodings/station_encodings.pkl
```

### `src/dataset.py` 추가 — `export_encodings()`

```python
def export_encodings(df: pd.DataFrame, train_end: str) -> None:
    """학습 기간 기준 station 인코딩 테이블 생성 및 저장"""
    # station_meta: csId → 유형·시도·시군구
    # station_encodings: csId → {station_target_mean, hour_means{}}
```

### config.py 추가

```python
ENCODINGS_DIR  = ROOT_DIR / "outputs" / "encodings"
HISTORY_DIR    = ROOT_DIR / "outputs" / "history"
HISTORY_FILE   = HISTORY_DIR / "station_history.csv"
REALTIME_DIR   = ROOT_DIR / "data" / "realtime"
```

---

## 설계 변경 이력

| 날짜 | 변경 내용 | 이유 | 결정자 |
|------|-----------|------|--------|
| 2026-06-01 | 초기 설계 + config·dataset·main 전면 수정 | 실제 데이터 기반 재설계 | 설계 담당 |
| 2026-06-01 | dataset.py 누수 수정: target_col → 모든 TARGET_COLS drop | 타겟 컬럼이 X에 포함되는 누수 발견 | 설계 담당 |
| 2026-06-01 | 시간 파생 피처 추가 (hour, dayofweek, is_weekend) | R² 개선 목적 | 설계 담당 |
| 2026-06-01 | 불균형 처리: 비영 샘플 가중치 (NONZERO_WEIGHT=9) | 타겟 0.0 비율 91.9% 대응 | 설계 담당 |
| 2026-06-01 | 가중치 전략 폐기 — 비영 과소예측 편향 확인 | rmse_nonzero 악화 (0.62→음수) | 설계 담당 |
| 2026-06-01 | is_daytime, station_target_mean, station_hour_mean 추가 | rmse_nonzero 개선 목적 | 설계 담당 |
| 2026-06-01 | v1.0 확정 (station×hour 2차 encoding까지) | 수확 체감 구간 도달, 구조적 한계 판단 | 설계 담당 |
| 2026-06-01 | v2.0 구현 — 2단계 모델 (분류기+회귀기) | rmse_nonzero baseline 돌파 목표 | 설계 담당 |
| 2026-06-01 | scale_pos_weight 제거, CLF_THRESHOLD=0.08 | 가중치 학습 불안정(3R early stopping) 확인 | 설계 담당 |
| 2026-06-01 | 평가 지표 정리 — overall RMSE·R² 제거 | 2단계 모델에서 의미 없음 | 설계 담당 |
| 2026-06-01 | v2.0 설계 — 2단계 모델 (분류기 + 회귀기) | rmse_nonzero baseline 0.38 돌파 목표 | 설계 담당 |
| 2026-06-01 | CLF_THRESHOLD=0.08 확정 | rmse_nonzero를 핵심 지표로 정의. overall RMSE 악화는 91.9% 零 분포에서 비영 구간 집중의 필연적 trade-off | 설계 담당 |
| 2026-06-02 | 데이터 확장: 1월 단일 → 1~12월 12개 파일 | 실제 운용 데이터 전체 반영 | 설계 담당 |
| 2026-06-02 | Split 기준일 변경: Train~10월말 / Val 11월 / Test 12월 | 12개월 데이터 기반 비율 재설정 (10:1:1) | 설계 담당 |
| 2026-06-02 | load_data() 인터페이스 변경: file_path → data_dir (glob 방식) | 월별 파일 12개 자동 병합 | 설계 담당 |
| 2026-06-02 | v2.1 — lag 피처 추가 (lag_1h, lag_2h, lag_24h) | 12개월 데이터로도 baseline 미돌파 → 시계열 맥락 부재가 원인 | 설계 담당 |
| 2026-06-02 | v3.0 — 목적 전환: 비율 회귀 → 만차 여부 이진 분류 | 실사용 목적이 만차 여부 판단임을 확인 | 설계 담당 |
| 2026-06-02 | v3.0 — 데이터 1개월(202501), split 1월 기준 복귀 | 빠른 검증 우선, 성능 확인 후 확장 결정 | 설계 담당 |
| 2026-06-02 | v3.0 — scale_pos_weight=24, F2 threshold 탐색 | Recall 우선 목표, 만차 3.92% 불균형 대응 | 설계 담당 |
| 2026-06-02 | v3.0 — scale_pos_weight 제거 | 1R 조기종료 학습 불안정 (v2.0과 동일 현상), threshold로만 대응 | 설계 담당 |
| 2026-06-02 | v3.0 — Negative Undersampling 1:5 + Recall≥0.85 threshold 탐색 | 극단적 Recall 우선 전략, 손실 함수 왜곡 없이 불균형 대응 | 설계 담당 |
| 2026-06-02 | v3.0 — threshold 기준 변경: Recall≥0.85 → F2 최대화 후 Recall≥0.85 복귀 | Precision 7%가 낮지만 Recall 우선 목표 재확인 | 설계 담당 |
| 2026-06-02 | v3.0 확정 — 데이터 12개월로 확장, split 10:1:1 | 1개월 모델 방향 확정 후 전체 데이터 학습 | 설계 담당 |
| 2026-06-04 | 실시간 추론 모듈 설계 — src/infer.py, encodings, history | cpStat≠1 점유, CSV 이력 저장 방식 확정 | 설계 담당 |
| 2026-06-04 | 실시간 입력 방식 확정 — CSV 파일 (data/realtime/) | predict_realtime_from_csv() 진입점 추가 | 설계 담당 |
| 2026-06-05 | 데이터 품질 수정 — 4월 ME#S 이상 ID 750개 제거 후 재학습 | 다른 달에 없는 ID 형식, 노이즈로 판단 | 설계 담당 |
| 2026-06-10 | PR-AUC 평가 지표 추가 | 불균형 데이터(3.92%)에서 ROC-AUC 과대평가 보완 | 설계 담당 |
