# Data Contract

이 문서는 딥러닝 프로젝트에서 사용할 입력/중간 산출물/예측 결과 형식을 정의합니다.

## 1. Protein FASTA

권장 파일명:

```text
data/private/h3n2_ha1_proteins.fasta
```

요구사항:

- FASTA header의 첫 토큰은 고유 `seq_id`
- sequence는 amino-acid alphabet 기준
- `X`, `B`, `Z`, `J`, `U`, `O`, `*`, `-`는 허용하되 invalid/ambiguous feature로 기록
- HA1 boundary가 적용된 서열인지 full HA인지 metadata에 명시

## 2. Metadata TSV

필수 컬럼:

```text
seq_id
subtype
gene
protein_region
collection_date
year
host
country
source
```

권장 컬럼:

```text
strain_name
clade
lineage
passage
vaccine_strain
reference_id
sequence_source
data_license
```

## 3. Label TSV

항원성 좌표 회귀용:

```text
seq_id
antigenic_x
antigenic_y
label_source
assay
```

Pairwise HI/distance 예측용:

```text
seq_id_a
seq_id_b
antigenic_distance
hi_titer_a_against_b
label_source
assay
```

분류용:

```text
seq_id
label_name
label_value
label_source
```

## 4. Manifest TSV

`prepare_protein_manifest.py`가 생성하는 기본 산출물입니다.

```text
seq_id
description
length
valid_aa_count
invalid_aa_count
ambiguous_aa_count
stop_count
gap_count
invalid_characters
aa_A
...
aa_Y
```

## 5. Model Prediction TSV

회귀:

```text
seq_id
split
target_antigenic_x
target_antigenic_y
pred_antigenic_x
pred_antigenic_y
prediction_model
```

Novelty/risk:

```text
seq_id
split
novelty_score
risk_class
prediction_model
top_explanatory_mutations
```

Mutation scan:

```text
reference_id
position
reference_aa
mutant_aa
mutation
delta_score
prediction_model
```
