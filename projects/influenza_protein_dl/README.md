# Influenza Protein DL Project

인플루엔자 단백질 서열 기반 딥러닝 모델 개발을 위한 작업 폴더입니다.

이 폴더는 기존 `influmatics` 패키지의 FASTA QC, alignment, mutation, translate,
antigenic/resistance scan 흐름 위에 단백질 언어모델(PLM) 기반 예측 레이어를
얹는 것을 목표로 합니다.

## Current Deliverables

- `BLUEPRINT.md`: 연구/모델 개발 청사진
- `TASK_SPLIT.md`: 사용자가 할 일과 Codex가 진행할 수 있는 일
- `DATA_CONTRACT.md`: 학습 데이터/라벨/예측 결과의 파일 형식
- `reports/decision_guide_from_package_v2.md`: 사용자 결정 가이드 적용 요약
- `configs/baseline.yaml`: 첫 실험 설정 템플릿
- `src/influenza_protein_dl/sequence_features.py`: 단백질 서열 검증 및 기초 feature
- `src/prepare_protein_manifest.py`: FASTA에서 manifest TSV를 만드는 시작 스크립트
- `tests/test_sequence_features.py`: feature 함수의 최소 테스트

## Suggested First Target

첫 모델은 범위를 좁혀 `A/H3N2 HA1 amino-acid sequence -> antigenic map coordinate
or antigenic novelty score`로 시작합니다. 이유는 현재 저장소가 HA/NA 변이,
항원 부위, numbering 문제를 이미 중심 과제로 두고 있고, 참고 논문에서도 H3 HA1
항원성 예측이 가장 직접적으로 연결되기 때문입니다.

## Quick Start

저장소 루트(`influmatics_code`)에서:

```bash
python projects/influenza_protein_dl/src/prepare_protein_manifest.py \
  --input path/to/proteins.fasta \
  --out analysis_results/protein_manifest.tsv
```

합성 예시 데이터로 전체 흐름을 확인하려면:

```bash
PYTHONPATH=projects/influenza_protein_dl/src \
python projects/influenza_protein_dl/src/prepare_protein_manifest.py \
  --input projects/influenza_protein_dl/examples/toy_ha1.fasta \
  --out analysis_results/influenza_protein_dl/toy_manifest.tsv

PYTHONPATH=projects/influenza_protein_dl/src \
python projects/influenza_protein_dl/src/validate_model_inputs.py \
  --manifest analysis_results/influenza_protein_dl/toy_manifest.tsv \
  --metadata projects/influenza_protein_dl/examples/toy_metadata.tsv \
  --labels projects/influenza_protein_dl/examples/toy_labels.tsv

PYTHONPATH=projects/influenza_protein_dl/src \
python projects/influenza_protein_dl/src/build_coordinate_dataset.py \
  --manifest analysis_results/influenza_protein_dl/toy_manifest.tsv \
  --metadata projects/influenza_protein_dl/examples/toy_metadata.tsv \
  --labels projects/influenza_protein_dl/examples/toy_labels.tsv \
  --out analysis_results/influenza_protein_dl/toy_dataset.tsv

PYTHONPATH=projects/influenza_protein_dl/src \
python projects/influenza_protein_dl/src/train_ridge_baseline.py \
  --dataset analysis_results/influenza_protein_dl/toy_dataset.tsv \
  --outdir analysis_results/influenza_protein_dl/toy_ridge \
  --split-strategy temporal \
  --train-end-year 2021 \
  --valid-years 2022 \
  --test-start-year 2023
```

기존 `influmatics translate`가 만든 AA mutation TSV가 있으면 항원 부위 feature도
붙일 수 있습니다.

```bash
PYTHONPATH=projects/influenza_protein_dl/src \
python projects/influenza_protein_dl/src/build_mutation_features.py \
  --aa-mutations projects/influenza_protein_dl/examples/toy_aa_mutations.tsv \
  --antigenic-sites projects/influenza_protein_dl/examples/toy_antigenic_sites.json \
  --out analysis_results/influenza_protein_dl/toy_mutation_features.tsv

PYTHONPATH=projects/influenza_protein_dl/src \
python projects/influenza_protein_dl/src/build_coordinate_dataset.py \
  --manifest analysis_results/influenza_protein_dl/toy_manifest.tsv \
  --metadata projects/influenza_protein_dl/examples/toy_metadata.tsv \
  --labels projects/influenza_protein_dl/examples/toy_labels.tsv \
  --mutation-features analysis_results/influenza_protein_dl/toy_mutation_features.tsv \
  --out analysis_results/influenza_protein_dl/toy_dataset_with_mutations.tsv
```

테스트:

```bash
PYTHONPATH=projects/influenza_protein_dl/src pytest projects/influenza_protein_dl/tests
```

## Data Rule

제한 데이터, 특히 GISAID 유래 FASTA/metadata는 커밋하지 않습니다. 원본 데이터는
`data/private/` 또는 저장소 밖에 두고, 이 폴더에는 스키마, 코드, 작은 공개 예시만
남깁니다.
