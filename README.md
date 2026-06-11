# EV 충전기 만차 예측 — LightGBM v3.0

EV 충전소의 **1시간 뒤 / 2시간 뒤 만차 여부**를 LightGBM 이진 분류로 예측하는 AI 파이프라인입니다.

---

## 프로젝트 구조

```
ev-charger-predict/
├── src/
│   ├── dataset.py      데이터 로드, 피처 엔지니어링, 시간 기반 split
│   ├── train.py        LightGBM 학습, 모델 저장/로드
│   ├── evaluate.py     메트릭 계산, 보고서용 시각화
│   ├── predict.py      저장된 모델로 배치 추론
│   └── infer.py        실시간 추론 파이프라인 (API 데이터 → 만차 예측)
├── outputs/
│   ├── models/         학습된 모델 (.pkl)
│   ├── plots/          피처 중요도 · 보고서 차트 (.png)
│   ├── metrics/        평가 결과 (.csv)
│   ├── encodings/      충전소 메타 · 인코딩 테이블 (.pkl, git 제외)
│   └── history/        실시간 lag 이력 (station_history.csv, git 제외)
├── data/
│   ├── processed/      ratio_dataset_202501.csv ~ 202512.csv (git 제외)
│   └── realtime/       실시간 API CSV 입력 (git 제외)
├── config.py           모든 하이퍼파라미터 & 경로 중앙 관리
├── main.py             학습 파이프라인 진입점
└── plot_report.py      보고서용 시각화 일괄 생성
```

---

## 데이터

| 항목 | 내용 |
|------|------|
| 파일 | `ratio_dataset_202501.csv` ~ `ratio_dataset_202512.csv` (월별 12개) |
| 기간 | 2025-01-01 ~ 2025-12-31 |
| 충전소 수 | 2,069개 |
| 타겟 | `1시간뒤_정답(y)`, `2시간뒤_정답(y)` → 만차(1.0) 여부 이진화 |
| 클래스 분포 | 만차 3.92% / 비만차 96.08% |
| 데이터 품질 | 4월 ME#S 접두 이상 ID 750개 로드 시 자동 제거 |

`data/processed/` 폴더에 CSV 파일을 넣어야 실행 가능합니다.

---

## Split 전략

랜덤 split을 사용하지 않습니다. 미래 데이터 누수 방지를 위해 **시간 기반 split**을 적용합니다.

| 구간 | 기간 | 비율 |
|------|------|------|
| Train | ~ 2025-10-31 23:00 | 10개월 |
| Validation | 2025-11-01 ~ 2025-11-30 23:00 | 1개월 |
| Test | 2025-12-01 ~ | 1개월 |

---

## 피처 목록 (22개)

| 구분 | 피처 | 비고 |
|------|------|------|
| 상태 | `현재상태`, `1시간전` | 원본 컬럼 |
| 시간 파생 | `hour`, `dayofweek`, `is_weekend`, `is_daytime` | 기준시간에서 추출 |
| Lag | `lag_1h`, `lag_2h`, `lag_24h` | 충전소별 shift, NaN은 LightGBM 자체 처리 |
| 충전소 속성 | `충전기개수`, `충전기구분`, `유형`, `시도`, `시군구` | |
| 환경 | `기온`, `강수량`, `공휴일` | |
| 지역 통계 | `지역_전기차수`, `지역_충전기수`, `충전기당_경쟁률` | |
| Target Encoding | `station_target_mean`, `station_hour_mean` | train 기간 기준, leakage 없음 |

---

## 불균형 처리

만차 비율 3.92%의 극심한 클래스 불균형을 **Negative Undersampling**으로 대응합니다.

- train set에서 negative를 positive의 5배로 random sampling (1:5)
- val / test는 원본 분포 유지 (실제 운용 환경 반영)
- threshold: val set에서 **Recall ≥ 0.85를 만족하는 최대값** 탐색

---

## 성능 결과 (v3.0, Test set: 2025-12월)

| 지표 | 1시간뒤 | 2시간뒤 |
|------|---------|---------|
| **Recall** | **0.862** | **0.862** |
| F2 | 0.393 | 0.393 |
| F1 | 0.198 | 0.198 |
| Precision | 0.113 | 0.113 |
| AUC | 0.876 | 0.876 |
| **PR-AUC** | **0.326** | **0.326** |
| threshold | 0.15 | 0.15 |

> PR-AUC 무작위 baseline ≈ 0.04 (클래스 비율). 본 모델은 baseline 대비 **8.2배** 달성.

---

## 피처 중요도 Top 5 (gain 기준)

| 순위 | 피처 | 1시간뒤 | 2시간뒤 |
|------|------|---------|---------|
| 1 | station_hour_mean | 55.5% | 55.6% |
| 2 | 충전기개수 | 19.0% | 19.1% |
| 3 | 현재상태 / lag_1h | 11.5% | 12.1% |
| 4 | 시군구 | 3.8% | 3.5% |
| 5 | station_target_mean | 2.4% | 2.2% |

---

## 실시간 추론

```python
from src.infer import predict_realtime_from_csv
from pathlib import Path

result = predict_realtime_from_csv(Path("data/realtime/charger_status.csv"))
# result: DataFrame[csId, proba_1h, full_1h, proba_2h, full_2h]
```

입력 CSV 컬럼: `csId`, `cpStat`, `statUpdateDatetime` (yyyyMMddHHmmss)

추론 시 `outputs/history/station_history.csv`에 현재 상태가 자동 누적되어 lag 피처로 활용됩니다.

---

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 학습 실행 (모델 + 인코딩 저장)
python main.py

# 보고서 시각화 생성
python plot_report.py
```

결과물은 `outputs/` 폴더에 저장됩니다.

---

## 설정 변경

`config.py`에서 모든 하이퍼파라미터와 경로를 관리합니다.

```python
FULL_CLF_NEG_RATIO: int = 5          # undersampling 비율
FULL_CLF_TARGET_RECALL: float = 0.85 # threshold 탐색 목표 Recall
LGBM_FULL_CLF_PARAMS = { "learning_rate": 0.05, "num_leaves": 31, ... }
```
