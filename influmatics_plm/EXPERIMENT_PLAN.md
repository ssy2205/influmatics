# H3N2 인플루엔자 과거 관측 자료 기반 PLM 항원성 거리 예측 및 항원 지도화 정밀 연구·실험 계획서
(Comprehensive Research & Experimental Protocol: Version 3.0)

> **문서 버전**: v3.0 (통계적 인과 분리 Ablation, 전향적 감시 분할, 다면적 평가 지표 전면 개정본)  
> **최종 갱신일**: 2026년 9월 8일  
> **대상 데이터**: Historical Influenza A(H3N2) HA1 Domain (1968~2023)  
> **주요 백본**: Pretrained Protein Language Model (ESM-2 650M)

---

## 1. 연구 개요 및 타깃 라벨의 수학적 정체성 정의

### 1.1 타깃 라벨(Target Label)의 엄밀한 수학적·생물학적 규정
선행 연구 및 모델 아키텍처 수립에 앞서, 예측하고자 하는 목적 변수(Target)를 혼선 없이 명확히 규정합니다.

* **본 연구의 타깃: 균주 쌍 간 상대적 혈청학적 항원성 거리 (Pairwise Antigenic Distance, $D_{AB}$)**
  - 본 연구는 단일 혈청 반응 역가(NHT: Normalized HI Titer)나 인위적인 2차원 투영 좌표 간 유클리드 거리가 아니라, **동일한 항혈청 패널에 대한 두 바이러스의 교차 반응성 차이를 요약한 대칭적 상대 거리($D_{AB}$)**를 1차 예측 타깃으로 설정합니다:
    $$D_{AB} = \frac{1}{|S_A \cap S_B|} \sum_{k \in S_A \cap S_B} \left| \log_2(H_{Ak}) - \log_2(H_{Bk}) \right|$$
  - **수학적 불변량**: 타깃이 두 바이러스 간의 비유사도(Dissimilarity)이므로, 모델은 **대칭성($D_{AB} = D_{BA}$)**과 **비음수성($D_{AB} \ge 0$)**을 만족해야 합니다.
  - **2D 항원 지도와의 위계 구분**: 2차원 항원 지도(Antigenic Cartography)는 예측 모델의 직접적인 회귀 타깃이 아니며, **예측된 전역 거리 행렬을 사후적으로 2차원 공간에 투영·가시화하는 '다운스트림 시각화 단계(Post-hoc Visualization)'**로 완전히 분리합니다.

### 1.2 연구 윤리 및 생물학적 안전 가이드라인
* **후향적 관측 자료 한정**: 학술지(*Frontiers in Microbiology*, 2024 / *Scientific Reports*, 2025)에 기 출판된 과거 54년간(1968~2023)의 기 관측 분리주 및 공인 페럿 HI 역가 측정값에 엄격히 한정됩니다.
* **기능 획득 및 인공 생성 배제**: 신규 인공 변이 서열 생성(Generative design)이나 면역 회피 변이체 선별·최적화(Evasion optimization)와 관련된 일체의 작업을 배제하고, 순수 관측 데이터에 대한 메트릭 회귀와 차원 축소 지도화만을 수행합니다.

---

## 2. 엄격한 2단계 통계적 절제 실험 (Two-Stage Ablation Study)

이전 설계의 한계(풀링, 상호작용 피처, 손실 함수가 동시에 바뀌어 성능 향상의 원인을 규명하지 못하던 Confounded Ablation)를 극복하기 위해, **2단계 직교 절제 실험**으로 분리합니다.

```text
[ Stage A: Pair Representation Architecture 확정 ]
  PLM(ESM-2 650M) + Global Mean Pooling 고정
  R0 (Scalar L2) vs R1 (대칭 고용량) vs R2 (비대칭 고용량)
                │
                ▼ (최적 아키텍처 1종 확정)
[ Stage B: Component-wise Ablation (독립 검증) ]
  P0: PLM Baseline (확정된 Pair Head)
  P1: P0 + 생물학적 채널 풀링 (Epitope & Key-site Channel Augmentation)
  P2: P0 + N-당화 변동 피처 (Gain / Loss 분리 반영)
  P3: P0 + 순위 보존 복합 손실 (Combined Loss, λ 튜닝)
                │
                ▼ (CV에서 유의미한 모듈만 선별 결합)
  P4: 최종 통합 모델 (Clinically/Surveillance-Optimized Final Model)
```

---

### 2.1 Stage A — 쌍 상호작용 표현(Pair Representation) 결정
동일한 파라미터 버짓(Parameter Budget, MLP 크기 통제) 하에서 서열 임베딩 상호작용 구조를 먼저 비교합니다:

| 모델 코드 | 입력 피처 형태 ($v$) | 입력 차원 | 수학적 성질 | 검증 질문 |
| :---: | :--- | :---: | :---: | :--- |
| **R0** | $\|u - v\|_2$ (스칼라 노름) | 1 | 유클리드 공리 보장 | "가장 단순한 기하학적 거리의 기저 성능은?" |
| **R1** | $[\|u - v\|, u \odot v]$ | $2d$ (2,560) | **엄격한 대칭성 보장** | "대칭성을 지키면서 상호작용을 줄 때의 성능은?" |
| **R2** | $[u, v, u - v, u \odot v]$ | $4d$ (5,120) | 비대칭 (방향성 허용) | "비대칭 자유 표현형을 허용할 때의 성능 상한선은?" |

* *참고*: R1에서 $(u-v)^2$은 정보적으로 $|u-v|$와 중복되므로 배제하고 $[|u-v|, u \odot v]$ 2개 블록으로 간소화합니다. Hidden dimension을 조정하여 R1과 R2의 MLP 파라미터 수를 동등하게 통제합니다.

---

### 2.2 Stage B — 생물학적·알고리즘적 모듈 독립 절제 실험
Stage A에서 선정된 최적 헤드(예: R1)를 고정하고, 각 모듈의 기여도를 독립적으로 측정합니다:

| 모델 | 적용 컴포넌트 | 연구 질문 (Ablation Question) |
| :---: | :--- | :--- |
| **P0** | **PLM Baseline** (Global Mean + 최적 Head) | "순수 ESM-2 650M 임베딩만의 순수한 예측 기저치는?" |
| **P1** | P0 + **채널 분리 생물학적 풀링** ($[z_{\text{global}}; z_{\text{epi}}; z_{\text{key}}]$) | "알려진 에피토프 및 핵심 잔기 영역의 정보가 유의미한 이득을 주는가?" |
| **P2** | P0 + **N-당화 Gain/Loss 보조 피처** | "에피토프 차폐를 유발하는 당화 자리 획득/소실 정보가 오차를 줄이는가?" |
| **P3** | P0 + **순위 보존 복합 손실** ($\text{SmoothL1} + \lambda \text{Ranking}$) | "순위 최적화 목적함수가 위험 변이 선별 순위(Spearman $\rho$)를 향상시키는가?" |
| **P4** | **최종 통합 모델** (P0 + CV에서 유의미했던 컴포넌트 결합) | "유효성이 입증된 핵심 요소들의 결합 시너지 효과는?" |

---

### 2.3 Classical Baseline 필수 포함 (Reviewer 대응)
"왜 거대 단백질 언어모델(ESM-2)을 써야 하는가?"를 학술적으로 증명하기 위해 고전 기준선을 함께 벤치마크합니다:
* **C0 (고전 서열 거리 모델)**: Hamming 거리 및 AAindex 물리화학적 특성 벡터 + Ridge / AdaBoost ([Li et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10834737/) 기준선)
* **C1 (PLM 선형 모델)**: Frozen ESM-2 임베딩 + Ridge 회귀 ([Durazzi et al., 2025](https://doi.org/10.1038/s41598-025-03275-2) 기준선)
* **P0 (PLM 비선형 모델)**: Frozen ESM-2 임베딩 + 샴 MLP Head

---

## 3. 생물학적 가이드 풀링의 정교화 (임의의 2~3배 가중치 폐기)

### 3.1 임의의 고정 가중치(`2×`, `3×`)의 폐기
* H3N2의 항원성 변이는 유행 시즌에 따라 핵심 잔기(140, 144, 145, 155, 156, 158, 159, 173, 186, 189, 208, 213 등)가 동적으로 달라지므로, 사람이 "에피토프 2배, 7개 위치 3배"와 같이 고정 상수를 부여하는 것은 학술적 타당성이 약합니다.

### 3.2 채널 분리형 다중 도메인 풀링 (Channel-wise Augmentation)
전체 HA1의 구조적 맥락을 손실 없이 보존하면서, 면역학적 중요 부위를 별도의 피처 채널로 병렬 제공합니다:
$$z = [z_{\text{global}} \;;\; z_{\text{epitope}} \;;\; z_{\text{key}}]$$
1. **$z_{\text{global}}$ (329개 전장 평균)**: HA1 전체 단백질의 글로벌 폴딩 및 보존적 프레임워크 표상
2. **$z_{\text{epitope}}$ (알려진 에피토프 평균)**: 5대 에피토프(A~E, 약 130개 잔기) 잔기들의 평균 임베딩
3. **$z_{\text{key}}$ (7대 역사적 전이 잔기 평균)**: Koel et al. (2013)의 7대 잔기(145, 155, 156, 158, 159, 189, 193번) 평균 임베딩
* *효과*: 임의의 스칼라 곱(`3×`)이 완전히 사라지며, 각 부위의 상대적 중요도는 후속 신경망 계층이 데이터로부터 스스로 가중치를 학습합니다.

### 3.3 H3 표준 넘버링 정렬 선행
* 자연 분리주 간 인델(삽입/결손)로 인한 인덱스 밀림을 방지하기 위해, 모든 잔기 추출은 **A/Aichi/2/1968 기준 H3 표준 넘버링 좌표계**에 사전에 정렬된 상태로만 수행합니다.

---

## 4. 실전 감시 현실을 반영한 시계열 분할 (Temporal Split Protocol)

### 4.1 '완전 Node-disjoint'의 한계 극복 및 2대 평가 분할
* 실제 백신 감시 현장에서는 **"새로 출현한 순수 신규 변이 바이러스"를 "과거에 이미 구축된 기지의 표준 항혈청(Reference Antisera)"과 교차 반응**시켜 평가합니다.
* 따라서 한쪽 노드(신규 바이러스)는 완전히 미지이지만, 다른 쪽 노드(참조 혈청/백신주)는 과거 데이터를 공유하는 것이 감시의 실제 목적입니다.

```text
[ Primary Evaluation: 실전 감시형 전향적 분할 (Surveillance Prospective Split) ]
  • Train 세트 (과거): Past Viruses × Past Reference Antisera
  • Test 세트 (미래): NEW Circulating Viruses × PAST Reference Antisera
  * 조건: 신규 피검 바이러스 노드 중복 = 0건 (완전 미지), 참조 백신주 노드 중복 = 허용

[ Secondary Evaluation: 엄격한 콜드 스타트 스트레스 테스트 (Strict Cold-Start Split) ]
  • Train 노드 ∩ Test 노드 = ∅ (양쪽 바이러스가 모두 처음 보는 균주인 가혹한 외삽 한계 테스트)
```

---

### 4.2 연도별 2대 벤치마크 평가 체계 (Primary vs Secondary)

모든 하이퍼파라미터(학습률, 드롭아웃, 랭킹 가중치 $\lambda$)는 **2003~2018년 개발 세트 내 3-Fold Rolling CV에서 완전히 확정(Fix)**하며, 2019~2022년 테스트셋은 절대 튜닝에 사용하지 않습니다.

#### ① Primary Benchmark: 점진적 연례 갱신 시뮬레이션 (Rolling Prospective Test)
매년 질병관리청 및 WHO 협력센터가 신규 데이터를 반영하여 모델 가중치를 점진적으로 재학습(Refit)한다고 가정하는 실전 감시 평가:
* $\text{Train } (\le 2018) \longrightarrow \text{Test } (2019\text{ 시즌})$
* $\text{Train } (\le 2019) \longrightarrow \text{Test } (2020\text{ 시즌})$
* $\text{Train } (\le 2020) \longrightarrow \text{Test } (2021\text{ 시즌})$
* $\text{Train } (\le 2021) \longrightarrow \text{Test } (2022\text{ 시즌})$

#### ② Secondary Benchmark: 장기 고정 외삽 감쇄 곡선 (Frozen Future Test)
한 번 배포된 모델이 재학습 없이 시간이 흐름에 따라 성능이 어떻게 저하(Decay)되는지 측정하는 장기 외삽 스트레스 테스트:
* $\text{Train } (\le 2018) \longrightarrow \text{Test } 2019 \text{ (1년 갭)} \to 2020 \text{ (2년 갭)} \to 2021 \text{ (3년 갭)} \to 2022 \text{ (4년 갭)}$

---

## 5. 다면적 평가 지표 체계 및 해석 가이드라인

단일 점수(Pooled MAE)가 특정 대규모 시즌에 의해 왜곡되는 것을 방지하기 위해, **시즌별 성능을 산출한 후 매크로 평균(Macro-averaged)**을 적용합니다.

### 5.1 종합 평가지표 일람

| 평가 역할 | 평가지표 | 공식 / 방법론 | 임상적·학술적 해석 기준 |
| :--- | :--- | :--- | :--- |
| **1. 절대 오차** | **Macro-MAE** | $\frac{1}{S}\sum_{s=1}^S \text{MAE}_s$ | 시즌별 불균형을 배제한 평균 절대 오차 ($< 0.60\text{ a.u.}$ 목표) |
| | **Macro-RMSE** | $\frac{1}{S}\sum_{s=1}^S \text{RMSE}_s$ | 치명적인 극단값 오차(Catastrophic Outlier) 억제 여부 |
| **2. 백신주 기준 순위** | **Per-Reference Spearman $\rho$** | $\frac{1}{|R|}\sum_{r \in R} \rho(y_{\text{serum } r}, \hat{y}_{\text{serum } r})$ | **임상적 최우선 지표**: 특정 백신 혈청 기준에서 신규 변이 간 상대적 회피 순위를 얼마나 잘 줄 세우는가 |
| **3. 드리프트 판별력** | **AUROC at 2 AU** | $D \ge 2.0\text{ a.u.}$ 기준 이진 분류 ROC 곡선하면적 | 4배 역가 저하(WHO 백신 업데이트 검토 기준) 변이를 선별하는 판별력 |
| | **Sensitivity / Specificity** | $D \ge 2.0\text{ a.u.}$ 임계점 기준 혼동행렬 | 실제 탈출 변이를 놓치지 않는 민감도 및 위양성 통제도 |
| **4. 2D 형상 위상** | **Procrustes Disparity ($M^2$)** | 앵커 삼각측량 맵 vs 실제 맵 정렬 잔차 제곱합 | 2차원 공간 상의 전역 클러스터 기하학적 배치 일치도 |
| | **Axis 1 Pearson $r$** | 1축(시간적 항원 표류 궤적) 피어슨 상관계수 | 연대순 항원 진화 경로의 올바른 전진 방향 복원력 ($r > 0.50$) |

---

## 6. N-당화(Glycosylation) 피처 및 앵커 삼각측량 사후 시각화

### 6.1 N-당화 보조 피처의 방향성 분리 (Gain / Loss)
단순 $\Delta \text{Gly}$ 스칼라 값 대신, 당화 위치의 획득과 소실을 2차원 벡터로 분리하여 제공합니다:
$$f_{\text{gly}} = [\text{Count}(\text{Glycosylation Gain}) \;,\; \text{Count}(\text{Glycosylation Loss})]$$
* *의미*: 에피토프 부위의 당사슬 추가(항체 차폐 유발)와 당사슬 결실(에피토프 노출)의 비대칭적 면역학적 효과를 모델이 구분하여 학습.

### 6.2 기준 백신주 앵커 기반 사후 삼각측량 (Post-hoc Landmark Cartography)
* **평가 시점 누수 원천 차단**: 
  - 예측 시점 $t$에서 사용하는 앵커 백신주는 반드시 $\text{Date}_{\text{anchor}} < t$인 과거 공인 백신주만 사용합니다.
  - A/Darwin/9/2021과 같은 최신 백신주는 2019~2022년 예측 평가가 완전히 끝난 후, 전체 진화 궤적을 통합 시각화하기 위한 **'사후 시각화(Post-hoc visualization only)'**로 엄격히 한정합니다.
* **삼각측량(Multilateration) 방식**:
  $$\min_{(x, y)} \sum_{k \in \text{Landmarks}} \left( \sqrt{(x - L_{kx})^2 + (y - L_{ky})^2} - \hat{D}(\text{신규 변이}, \text{백신주 } k) \right)^2$$
  - 자유 MDS의 고질적 문제인 180도 축 반전 및 회전 모호성을 완전히 해소하여 Procrustes Disparity를 획기적으로 낮춥니다.

---

## 7. 도구별 분업 실행 파이프라인 (Execution Plan)

### 7.1 코덱스 (Codex) — 순수 알고리즘 단위 모듈 작성 (안전 필터 무관 인터페이스)
1. **`scripts/rolling_temporal_split.py`**:
   - 2003~2018 데이터에 대해 New virus는 완전 격리하되 Historical reference는 허용하는 Prospective Split 및 3-Fold Rolling CV 생성기 구현.
2. **`scripts/combined_ranking_loss.py`**:
   - `SmoothL1Loss` + $\lambda \times$ `MarginRankingLoss` 모듈에 $\lambda$ 가중치 인자 반영.
3. **`scripts/landmark_multilateration.py`**:
   - 기준 좌표가 고정된 Landmark 점들과의 거리로부터 신규 노드의 2D 좌표를 최적화하는 삼각측량 솔버 구현.

### 7.2 안티그래비티 (Antigravity) — 전처리 및 훈련 자동화 파이프라인
1. **`scripts/02_preprocess_stages.py`**:
   - H3 표준 넘버링 기준 에피토프/핵심 잔기 마스킹 인덱스 추출 및 당화 Gain/Loss 피처 생성.
2. **`scripts/run_ablation_experiments.py`**:
   - Stage A (R0, R1, R2) $\to$ Stage B (P0, P1, P2, P3, P4) 순차 자동 학습 및 Rolling Prospective / Frozen Future 2대 벤치마크 지표 집계.

### 7.3 제미나이 스파크 (Gemini Spark) — 총괄 통계 검증 및 최종 보고서 배포
1. **통계적 가설 검정**:
   - P0 대비 각 모듈(P1, P2, P3, P4) 간 유의미성 검정(Paired t-test / Wilcoxon signed-rank test).
2. **최종 학술 보고서 배포**:
   - [influenza_plm_antigenic_mapping_report.html](file:///Users/shiftyellow/Documents/project-local/influmatics_project/influenza_plm_antigenic_mapping_report.html)에 개정된 2단계 절제 실험표, 2대 시계열 벤치마크 감쇄 곡선, 앵커 삼각측량 2D 지도 렌더링을 집대성하여 최종본 배포.
