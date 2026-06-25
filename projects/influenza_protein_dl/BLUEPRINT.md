# 인플루엔자 단백질 서열 딥러닝 모델 개발 청사진

## 1. 프로젝트 파일에서 읽은 현재 위치

현재 `influmatics`는 계절성 인플루엔자 서열 분석 프로그램으로 정리되고 있습니다.
문서상 MVP는 `FASTA input -> QC -> MAFFT alignment -> mutation table ->
Nextclade clade -> antigenic-site mutation detection -> antiviral marker scan ->
report/Streamlit UI` 흐름입니다.

코드에서는 이미 다음 기반이 있습니다.

- `influmatics.io`, `qc`: FASTA/FASTQ/CSV/TSV 입력과 기본 QC
- `alignment`: MAFFT wrapper
- `mutations`: reference 기반 nucleotide mutation table
- `translate`: nucleotide alignment에서 amino-acid mutation table 생성
- `numbering`: reference position과 alignment coordinate mapping
- `antigenic`, `resistance`: amino-acid coordinate mutation table을 받아 marker scan
- `data/markers`: H3N2 antigenic site, antiviral marker, numbering table scaffold

즉, 딥러닝 모델은 독립 앱으로 새로 만들기보다 기존 분석 파이프라인의 산출물에
붙는 예측 모듈로 설계하는 편이 가장 자연스럽습니다.

## 2. 참고 논문에서 얻는 설계 방향

프로젝트 내 PDF를 앞부분 중심으로 확인했습니다.

- influenza primer/antibody response 논문: HA와 NA가 주요 항원성 표면 단백질이며,
  HA head/stalk, NA, subtype, antigenic drift/shift, 백신 strain matching이 중요합니다.
- Durazzi et al. 2025: human influenza A(H3) HA1 서열에서 antigenic map coordinate,
  substitution impact, antigenic novelty를 예측하는 문제를 다룹니다. BiLSTM/ProtBERT
  같은 언어모델 계열이 세밀한 단일 아미노산 변화 평가에서 강점이 있었습니다.
- Ito et al. 2025 CoVFit: ESM-2 기반 PLM을 바이러스 fitness label에 맞게 적응시키고,
  미래 변이 fitness ranking과 single mutation effect 탐색에 사용했습니다.
- ProtGPT2 논문: 단백질 생성 모델의 가능성을 보여주지만, 현재 프로젝트의
  surveillance/antigenic prediction 목표에는 생성보다 예측/랭킹 모델이 우선입니다.

## 3. 추천하는 첫 연구 질문

1차 목표:

> A/H3N2 HA1 아미노산 서열만으로 항원성 좌표 또는 항원성 novelty score를 예측한다.

2차 확장:

- `HA1 sequence + metadata(year, clade)`로 antigenic map coordinate 회귀
- reference/vaccine strain 대비 antigenic distance 예측
- single amino-acid mutation을 넣은 in silico deep mutational scan
- NA/M2 단백질에 대해 antiviral resistance marker/phenotype 예측
- 장기적으로 HA/NA multi-protein viral fitness 또는 vaccine mismatch risk 예측

첫 목표를 H3N2 HA1으로 좁히는 이유:

- 현재 marker scaffold가 H3N2 HA antigenic site를 전제로 합니다.
- 참고 PLM 논문과 가장 잘 맞습니다.
- HA1은 항원성 drift 분석에서 해석 가능성이 높습니다.
- 제한된 팀/컴퓨팅 환경에서도 baseline -> PLM embedding 순서로 단계적 진행이 가능합니다.

## 4. 전체 파이프라인

```text
raw nucleotide/protein FASTA + metadata
-> influmatics QC / validation
-> segment/subtype/gene 확인
-> alignment + reference coordinate mapping
-> protein sequence 또는 AA mutation table 생성
-> dataset manifest 작성
-> temporal/clade-aware train/valid/test split
-> baseline feature 생성
-> PLM embedding 생성
-> supervised prediction head 학습
-> mutation effect / antigenic novelty ranking
-> report + Streamlit integration
```

## 5. 모델 단계

### Phase A: 규칙 기반/비딥러닝 baseline

- 입력: HA1 protein sequence, reference 대비 AA mutation table
- feature: AA composition, length, invalid AA count, antigenic-site mutation count,
  k-mer/hash feature, clade/year metadata
- 모델: linear/ridge regression, random forest, gradient boosting
- 목적: 데이터 누수, split, label 품질, 평가 기준을 먼저 검증

### Phase B: frozen PLM embedding

- 입력: HA1 amino-acid sequence
- backbone 후보: ESM-2, ProtBERT
- 방법: backbone은 고정하고 sequence embedding을 추출한 뒤 작은 MLP/regression head 학습
- 장점: 작은 데이터에서도 과적합 위험이 낮고 재현성이 높음

### Phase C: supervised fine-tuning

- 입력: HA1 sequence 또는 reference/mutant pair
- 출력:
  - antigenic coordinate `(x, y)` 회귀
  - vaccine/reference strain 대비 antigenic distance 회귀
  - novelty/risk class 분류
  - single mutation effect score
- 조건: 충분한 labeled antigenic/HI/map 데이터 확보 후 진행

### Phase D: 해석 및 surveillance

- 단일 변이 scan: 모든 가능한 HA1 single substitution을 넣고 risk score 변화 계산
- site-level aggregation: antigenic site A-E, receptor binding site, glycosylation motif
- 리포트: 새 sequence가 들어왔을 때 clade, 주요 mutation, novelty score, marker hit를 함께 표시

## 6. 데이터와 라벨

필수 입력:

- protein FASTA: HA1부터 시작, 이후 HA full length/NA/M2 확장
- metadata TSV: `seq_id`, `collection_date`, `year`, `subtype`, `segment`, `host`,
  `country`, `clade`, `source`
- reference 정보: subtype/gene별 reference sequence, CDS/HA1 boundary, numbering scheme

권장 라벨:

- antigenic map coordinate `(x, y)` 또는 strain-level antigenic cluster
- HI assay 기반 pairwise antigenic distance
- vaccine strain match/mismatch label
- phenotype resistance label이 있다면 NA/M2 확장에 사용

주의:

- GISAID 유래 데이터는 저장소에 커밋하지 않습니다.
- 같은 strain/near-duplicate가 train/test에 동시에 들어가지 않도록 cluster 또는 date split을 씁니다.
- random split보다 temporal split이 surveillance 목표에 맞습니다.

## 7. 평가 설계

회귀:

- RMSE/MAE for antigenic coordinates
- pairwise antigenic distance correlation
- future-year test set 성능

분류/랭킹:

- AUROC/AUPRC for novelty or mismatch
- top-k mutation effect enrichment
- vaccine update 후보 strain ranking 품질

생물학적 sanity check:

- known antigenic-site mutations가 높은 impact를 받는지
- glycosylation motif 변화가 score에 반영되는지
- clade/year만 외운 모델이 아닌지 metadata ablation으로 확인

## 8. 리스크와 대응

- numbering/coordinate 오류: 현재 저장소 TODO와 동일하게 가장 큰 리스크입니다.
  AA coordinate table과 reference boundary를 먼저 고정해야 합니다.
- label 부족: antigenic map/HI 데이터가 없으면 supervised target이 약합니다.
  먼저 mutation/site novelty proxy와 public labels로 baseline을 만듭니다.
- leakage: 유사 서열이 많아 random split 성능이 부풀 수 있습니다.
  temporal split과 sequence identity cluster split을 병행합니다.
- 컴퓨팅: fine-tuning은 GPU가 필요할 수 있습니다.
  frozen embedding부터 시작하면 CPU/작은 GPU에서도 진행 가능합니다.

## 9. 첫 4주 실행안

1주차:

- HA1/H3N2를 1차 범위로 확정
- 데이터 스키마 확정
- protein manifest 생성
- reference/numbering/HA1 boundary 정리

2주차:

- baseline feature 생성
- temporal split 구현
- dummy/proxy label로 end-to-end 학습 코드 뼈대 확인

3주차:

- antigenic map 또는 HI label 수집/정리
- baseline 모델 학습 및 leakage check
- report template 작성

4주차:

- ESM-2/ProtBERT frozen embedding 실험
- baseline과 비교
- single mutation scan prototype
- Streamlit/HTML report에 novelty score 연결
