# 서울시 약 수요예측 및 재고관리 대시보드
### 머신러닝 기반 약품 수요예측 및 재고 시뮬레이션 프로젝트

---

## INDEX

- [프로젝트 소개](#프로젝트-소개)
- [프로젝트 구조](#프로젝트-구조)
- [모델 개발 과정](#모델-개발-과정)
- [Streamlit 화면 구성](#streamlit-화면-구성)
- [Hugging Face 업로드 파일](#hugging-face-업로드-파일)
- [마지막 한마디](#마지막-한마디)

---

## 프로젝트 소개

### 서울시 약품 수요예측 및 재고관리 서비스

- 서울시 25개 구의 월별 약품 수요 데이터를 기반으로 2026년 수요를 예측합니다.
- 기상 데이터와 과거 수요 데이터를 결합하여 약품별, 구별 수요 흐름을 분석합니다.
- 여러 머신러닝 모델과 학습방법을 비교하여 예측 성능을 확인합니다.
- 가상 재고 CSV를 업로드하면 예측 수요 기반 권장 구매량을 계산합니다.

### 목표

- 약품별 월별 수요예측 결과 생성
- 2025년 실제 수요와 예측 수요 비교
- 학습모델 및 학습방법별 성능 비교
- 재고 기반 권장 구매량 계산
- Hugging Face Spaces에서 실행 가능한 대시보드 구현

---

## 프로젝트 구조

GitHub 저장소 기준 파일 구조입니다.

```text
Drug_main/
├── README.md                                      # 프로젝트 설명 문서
├── requirements.txt                               # 실행 패키지 목록
├── drug_demand_forecast_colab.ipynb               # Colab 학습 및 분석 노트북
├── drug_demand_train.py                           # VSCode/로컬 학습 스크립트
├── drug_demand_app.py                             # Streamlit 대시보드 파일
├── drug_demand_streamlit_app.py                   # Streamlit 앱 백업 파일
│
├── model_forecast_2026_all_methods.csv            # 2026년 전체 모델 예측 결과
├── model_comparison_2025_metrics.csv              # 2025년 검증 성능 지표
├── model_comparison_2025_predictions.csv          # 2025년 실제 수요 vs 예측 수요
│
├── baseline_best_model_2026_demand_forecast.csv   # 기준 최적 모델 2026년 예측 결과
├── baseline_models_2025_test_predictions.csv      # 기준 모델 2025년 검증 예측 결과
├── baseline_models_2025_validation_metrics.csv    # 기준 모델 2025년 검증 성능
│
├── random_forest_2026_demand_forecast.csv         # 호환용 2026년 예측 결과
├── random_forest_2025_test_predictions.csv        # 호환용 2025년 검증 예측 결과
├── random_forest_2025_validation_metrics.csv      # 호환용 2025년 검증 성능
│
├── data/
│   ├── flu_cleaned.csv                            # 해열진통소염제 수요 데이터
│   ├── hist_cleaned.csv                           # 항히스타민제 수요 데이터
│   ├── cacp_cleaned.csv                           # 진해거담제 수요 데이터
│   └── temp.csv                                   # 월별 기상 데이터
│
└── inventory_data/
    ├── virtual_cacp_hist_flu_3mon_only.csv        # 3개월 단위 예시 재고 데이터
    ├── virtual_inventory_3mo_totals.csv           # 전체 합계 기준 예시 재고 데이터
    └── virtual_inventory_3mon_direct.csv          # 구별 직접 입력 예시 재고 데이터
```

---

## 모델 개발 과정

### Phase 1. 데이터 전처리

**방식**

<details>
<summary>수요 데이터와 기상 데이터 결합</summary>

1. `flu_cleaned.csv`를 해열진통소염제 데이터로 사용  
2. `hist_cleaned.csv`를 항히스타민제 데이터로 사용  
3. `cacp_cleaned.csv`를 진해거담제 데이터로 사용  
4. `temp.csv`의 월별 기상 데이터를 수요 데이터와 병합  
5. 날짜, 약품구분, 구 이름, 수량 데이터를 학습 가능한 형태로 정리  

</details>

**생성 변수**

| 구분 | 변수 |
|---|---|
| 날짜 변수 | year, month, quarter |
| 계절성 변수 | month_sin, month_cos |
| 기상 변수 | avg_temp, max_temp, min_temp, rainfall |
| 파생 변수 | temp_range, cold_index |
| 과거 수요 변수 | qty_lag1, qty_lag2, qty_lag3, qty_lag12 |
| 이동평균 변수 | qty_ma3, qty_ma6 |

---

### Phase 2. 기준 모델 학습

**방식**

<details>
<summary>기준 학습모델 4종 비교</summary>

1. 랜덤포레스트  
2. XGBoost  
3. 라쏘회귀  
4. 릿지회귀  

</details>

**검증 기준**

- 학습 데이터: 2025년 이전 데이터
- 검증 데이터: 2025년 데이터
- 평가 지표: MAE, RMSE, MAPE, R2

---

### Phase 3. 학습방법별 모델 비교

**방식**

<details>
<summary>모델별 학습방법 확장</summary>

1. 그리드서치  
2. 옵튜나  
3. 보팅  
4. 배깅  
5. 부스팅  
6. 스태킹  

</details>

**학습 조합**

| 학습모델 | 학습방법 |
|---|---|
| 랜덤포레스트 | 그리드서치, 옵튜나, 보팅, 배깅, 부스팅, 스태킹 |
| XGBoost | 그리드서치, 옵튜나, 보팅, 배깅, 부스팅, 스태킹 |
| 라쏘회귀 | 그리드서치, 옵튜나, 보팅, 배깅, 부스팅, 스태킹 |
| 릿지회귀 | 그리드서치, 옵튜나, 보팅, 배깅, 부스팅, 스태킹 |
| 혼합앙상블 | Ridge+RandomForest+XGBoost, Lasso+Ridge+XGBoost, Ridge+Lasso+RandomForest+XGBoost |

---

### Phase 4. 재고 시뮬레이션

**방식**

<details>
<summary>예측 수요 기반 권장 구매량 계산</summary>

1. 사용자가 재고 CSV를 업로드합니다.  
2. 현재 재고와 다음 달 예측 수요를 비교합니다.  
3. 안전 여유율을 반영하여 권장 구매량을 계산합니다.  
4. 구매 후 재고와 월말 재고를 월별로 갱신합니다.  

</details>

**재고 계산 흐름**

```text
월초재고
→ 권장구매수량 추가
→ 구매후재고
→ 예측수요 차감
→ 월말재고
→ 다음 달 월초재고
```

---

## Streamlit 화면 구성

### 검증 비교

- 2025년 검증 성능 표 확인
- 모델별 MAPE 비교
- 실제 수요와 예측 수요 그래프 확인

### 2026 예측 추세

- 월별 예측 수요 그래프 확인
- 학습모델, 학습방법, 약품구분, 구 단위 필터링

### 재고 시뮬레이션

- 재고 CSV 업로드
- 안전 여유율 설정
- 권장 구매량 및 월말 재고 확인

### 히트맵

- 구별, 월별, 약품별 예측 수요 확인
- 수요가 높은 지역과 시점 파악

---

## Hugging Face 업로드 파일

```text
streamlit_app.py
requirements.txt
drug_demand_models.pkl
model_forecast_2026_all_methods.csv
model_comparison_2025_metrics.csv
model_comparison_2025_predictions.csv
```

### 실행 방법

로컬 실행 시:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 재고 CSV 예시

```csv
약품구분,시군구명칭,현재재고
해열진통소염제,강남구,100000
항히스타민제,강남구,80000
진해거담제,강남구,90000
```

전체 합계 기준으로 입력할 수도 있습니다.

```csv
약품구분,시군구명칭,현재재고
전체합계,전체합계,5000000
```

---


## 마지막 한마디

| 이름 | <nobr>역할<nobr> | 느낀점 |
|:---:|:---:|:---|
| <nobr>유영찬<nobr> | <nobr>팀장<nobr> | 의약품 수요예측 프로젝트를 진행하며 데이터 분석과 머신러닝 모델 적용 과정을 직접 경험할 수 있었다. 다양한 변수와 모델을 비교하며 예측 성능을 향상시키는 과정의 중요성을 배울 수 있었다. 이번 프로젝트를 통해 실제 문제 해결 능력과 인공지능 활용 역량을 키울 수 있는 뜻깊은 경험이 되었다. |
| <nobr>강민식<nobr> | <nobr>팀원<nobr> | 약 수요예측 프로젝트를 진행하면서 데이터 전처리, 모델 비교, Streamlit 대시보드 구현까지 전체 흐름을 경험할 수 있었습니다. 단순히 예측 결과를 만드는 것에서 끝나지 않고 재고 시뮬레이션까지 연결하면서 실제 활용 가능성을 고민해볼 수 있어 의미 있었습니다. |
| <nobr>조예연<nobr> | <nobr>팀원<nobr> | 이번 프로젝트를 하면서 데이터들을 모델로 돌릴때 많은 변수를 통해 새롭게 알게 된 사실도 있었고 그로인해 많이 부족한 점들을 느꼈습니다. 그래도 미래예측수량과 재고 데이터를 만드는 방법을 알게되는 시간이었습니다. |

---
