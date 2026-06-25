# 해야 할 일 구분

## 사용자가 결정/수행해야 할 일

- 1차 타깃 확정: `H3N2 HA1 antigenic prediction`으로 시작할지, NA/M2 resistance까지
  동시에 볼지 결정
- 사용할 데이터 출처와 권한 확인: GISAID, NCBI, IRD, 논문 supplementary data 등
- 제한 데이터 관리 원칙 확정: 원본 FASTA/metadata를 저장소 밖에 둘지, `data/private/`에 둘지 결정
- 라벨 확보: antigenic map coordinate, HI assay table, vaccine strain match label,
  resistance phenotype 중 어떤 target을 쓸지 선택
- reference strain/CDS/HA1 boundary 확인: subtype/gene별 표준 reference와 numbering scheme 검토
- 실험 자원 확인: GPU 사용 가능 여부, 로컬/서버/Colab 중 어디서 학습할지 결정
- 생물학적 검증 기준 제시: 어떤 mutation/site가 모델 해석에서 반드시 확인되어야 하는지 선정

## Codex가 진행할 수 있는 일

- 프로젝트 폴더 구조 생성 및 문서화
- 데이터 스키마와 manifest 생성 스크립트 작성
- FASTA/metadata validation 코드 작성
- 기존 `influmatics` QC/translate/mutation 출력과 딥러닝 입력을 연결
- baseline feature 생성: AA composition, length, k-mer/hash, antigenic-site mutation count
- train/valid/test split 코드 작성: random, temporal, clade-aware split
- baseline 모델 학습 코드 작성
- PLM embedding 추출 코드 작성: ESM-2/ProtBERT 후보
- supervised prediction head 학습 코드 작성
- 평가 리포트 생성: metric table, prediction TSV, mutation effect ranking
- 테스트 작성 및 실행
- Streamlit 또는 HTML report에 모델 결과를 붙이는 작업

## Codex가 이미 시작한 일

- `projects/influenza_protein_dl/` 작업 폴더 생성
- 연구 청사진 작성
- 역할 분담표 작성
- 데이터 계약 문서 작성
- 단백질 서열 feature extraction 코드 작성
- FASTA manifest 생성 스크립트 작성
- manifest/metadata/label 검증 스크립트 작성
- antigenic coordinate dataset builder 작성
- temporal/random split 모듈 작성
- numpy 기반 ridge baseline 학습/평가 스크립트 작성
- 합성 예시 데이터 추가
- 최소 단위 테스트 작성

## 다음에 Codex가 바로 이어서 할 수 있는 일

- 실제 단백질 FASTA 예시를 받으면 manifest 생성 및 QC 요약 실행
- metadata TSV가 있으면 join/validation 스크립트 작성
- antigenic label 파일이 있으면 baseline 학습용 dataset builder 작성
- 실제 label dataset으로 baseline metric 산출
- 기존 `influmatics translate/antigenic` 결과에서 antigenic-site mutation count feature 연결
- GPU/네트워크 가능 환경이 정해지면 PLM embedding 추출 스크립트 작성
