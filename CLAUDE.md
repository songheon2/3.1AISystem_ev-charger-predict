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

## 코딩 규칙

| 규칙 | 내용 |
|------|------|
| 상수·하이퍼파라미터 | `config.py`에서만 관리, 하드코딩 금지 |
| 타입 힌트 | 모든 함수 시그니처 필수 |
| 로깅 | `print` 대신 `logging` 모듈 |
| 시드 | `SEED = 42` |
| 스타일 | 순수 스크립트, Jupyter 셀 방식 금지 |

---

## 설계 변경 이력

| 날짜 | 변경 내용 | 이유 | 결정자 |
|------|-----------|------|--------|
| 2026-06-01 | 초기 설계 + config·dataset·main 전면 수정 | 실제 데이터 기반 재설계 | 설계 담당 |
| 2026-06-01 | dataset.py 누수 수정: target_col → 모든 TARGET_COLS drop | 타겟 컬럼이 X에 포함되는 누수 발견 | 설계 담당 |
| 2026-06-01 | 시간 파생 피처 추가 (hour, dayofweek, is_weekend) | R² 개선 목적 | 설계 담당 |
| 2026-06-01 | 불균형 처리: 비영 샘플 가중치 (NONZERO_WEIGHT=9) | 타겟 0.0 비율 91.9% 대응 | 설계 담당 |
