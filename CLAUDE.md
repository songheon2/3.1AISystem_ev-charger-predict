# EV 충전기 예측 시스템 — 설계 명세

## 프로젝트 개요

- **데이터**: `ratio_dataset_202501.csv` (2025-01-01 ~ 2025-01-31, 1,539,336행 × 17컬럼)
- **충전소 수**: 2,069개
- **목표**: 1시간 뒤 / 2시간 뒤 충전 비율 예측 (회귀)
- **모델**: LightGBM

---

## 프로젝트 구조

```
ev-charger-predict/
├── src/
│   ├── __init__.py
│   ├── dataset.py      # 데이터 로드, 피처 엔지니어링, 시간 기반 split
│   ├── train.py        # LightGBM 학습, 모델 저장/로드
│   ├── evaluate.py     # 메트릭 계산, 플롯 저장
│   └── predict.py      # 추론
├── outputs/
│   ├── models/         # lgbm_*.pkl
│   ├── plots/          # fi_*.png
│   └── metrics/        # metrics_*.csv
├── data/processed/     # ratio_dataset_202501.csv
├── config.py           # 모든 상수·하이퍼파라미터
├── main.py             # 메인 파이프라인
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
| 타겟 분포 | 0.0이 약 91.9% (극도 불균형) |

---

## Split 전략

- **방식**: 시간 기반 split (랜덤 split 금지 — 미래 누수 방지)
- Train: `~ 2025-01-24 23:00`
- Val: `2025-01-25 00:00 ~ 2025-01-27 23:00`
- Test: `2025-01-28 00:00 ~`

---

## 피처 엔지니어링

- `기준시간` → `hour`, `dayofweek`, `is_weekend` 파생 후 원본 drop
- `CAT_COLS` → `category` dtype 변환 (LightGBM 자동 인식)
- 비영 샘플 가중치: `config.NONZERO_WEIGHT` 참조 (기본값 9 → 비영 가중치 10, 영 가중치 1)

---

## 모듈별 역할 및 인터페이스

### `config.py`
- 모든 상수·하이퍼파라미터 중앙 관리

### `src/dataset.py`
- `load_data(file_path) -> DataFrame`: CSV 로드, `기준시간` datetime 변환
- `split_data(df, target_col) -> (X_train, X_val, X_test, y_train, y_val, y_test)`: 피처 엔지니어링 + 시간 기반 split

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
