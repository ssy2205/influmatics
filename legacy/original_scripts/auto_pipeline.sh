#!/bin/bash
set -euo pipefail

# --- 기본값 설정 ---
# 입력 파일/폴더는 비워둡니다. 사용자가 반드시 지정해야 합니다.
READS=""
INDIR=""
# 레퍼런스와 Medaka 모델은 자주 바뀌지 않으므로 기본값을 지정해둘 수 있습니다.
REF="h3n2.fasta" # 여기에 고정할 레퍼런스 파일 경로를 적어두세요.
MODEL="r1041_e82_400bps_sup_v4.2.0"
THREADS=16
BREADTH_MIN=95
DEPTH_MIN=100
MAPQ_MIN=20      # 신뢰도 높은 매핑만 남기기 위한 MAPQ 필터 추가
OUTDIR="analysis_results"

usage() {
  cat <<EOF
Usage: $0 --ref reference.fa [options] (--reads single.fq | --indir fq_folder)

Required:
  --ref        Reference assembly FASTA
  And ONE of:
  --reads      Single ONT raw reads FASTQ(.gz) to process
  --indir      Directory containing multiple FASTQ(.gz) files to process

Options:
  --model      Medaka model (default: ${MODEL})
  --threads    Threads for tools (default: ${THREADS})
  --breadth    Min breadth percent (default: ${BREADTH_MIN})
  --depth      Min mean depth X (default: ${DEPTH_MIN})
  --mapq       Min mean mapping quality (default: ${MAPQ_MIN})
  --outdir     Main output directory (default: ${OUTDIR})
EOF
  exit 1
}

# --- 사용자 입력 옵션 처리 ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reads)   READS="$2"; shift 2;;
    --indir)   INDIR="$2"; shift 2;;
    --ref)     REF="$2"; shift 2;;
    --model)   MODEL="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --breadth) BREADTH_MIN="$2"; shift 2;;
    --depth)   DEPTH_MIN="$2"; shift 2;;
    --mapq)    MAPQ_MIN="$2"; shift 2;;
    --outdir)  OUTDIR="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "[ERR] Unknown arg: $1"; usage;;
  esac
done

# --- 입력값 유효성 검사 ---
if [[ -z "${REF}" || (! -f "${REF}") ]]; then
  echo "[ERR] Reference FASTA not provided or not found: --ref ${REF}"; usage;
fi
if [[ -n "${READS}" && -n "${INDIR}" ]]; then
  echo "[ERR] Please provide either --reads OR --indir, not both."; usage;
fi
if [[ -z "${READS}" && -z "${INDIR}" ]]; then
  echo "[ERR] Please provide either --reads OR --indir."; usage;
fi

# --- 필요 프로그램 확인 ---
need() { command -v "$1" >/dev/null 2>&1 || { echo "[ERR] '$1' not found in PATH"; exit 127; }; }
need minimap2; need samtools; need medaka_consensus; need seqkit

# ==============================================================================
# ===      하나의 샘플을 처리하는 핵심 파이프라인 함수      ===
# ==============================================================================
run_pipeline() {
  local fastq_file="$1"
  local sample_outdir="$2"
  local sample_name
  sample_name=$(basename "${fastq_file}" | sed -E 's/\.fq\.gz$|\.fastq\.gz$|\.fq$|\.fastq$//')

  echo "=================================================="
  echo ">>> Starting pipeline for sample: ${sample_name}"
  echo ">>> Output directory: ${sample_outdir}"
  echo "=================================================="

  mkdir -p "${sample_outdir}"/{logs,work}
  local LOG="${sample_outdir}/logs/run.log"
  
  # 함수 내에서 실행되는 모든 로그를 해당 샘플의 로그 파일에 저장
  (
    echo "=== Params for ${sample_name} ==="
    echo "FASTQ=${fastq_file}"
    echo "REF=${REF}"
    echo "MODEL=${MODEL}"
    echo "THREADS=${THREADS}"
    echo "BREADTH_MIN=${BREADTH_MIN}%"
    echo "DEPTH_MIN=${DEPTH_MIN}X"
    echo "MAPQ_MIN=${MAPQ_MIN}"
    echo "================================="

    local BAM="${sample_outdir}/work/aln.bam"
    if [[ ! -f "${BAM}" ]]; then
      echo "[*] Aligning reads..."
      minimap2 -t "${THREADS}" -x map-ont -a "${REF}" "${fastq_file}" \
        | samtools sort -@ "$((THREADS/2))" -o "${BAM}"
      samtools index "${BAM}"
    fi

    local COVTSV="${sample_outdir}/work/cov.tsv"
    echo "[*] Calculating coverage..."
    samtools coverage -o "${COVTSV}" "${BAM}"

    local KEEP_LIST="${sample_outdir}/work/keep_contigs.txt"
    echo "[*] Filtering contigs: breadth>=${BREADTH_MIN}%, depth>=${DEPTH_MIN}X, mapq>=${MAPQ_MIN}"
    awk -v bmin="${BREADTH_MIN}" -v dmin="${DEPTH_MIN}" -v mqmin="${MAPQ_MIN}" 'BEGIN{FS=OFS="\t"} NR>1 {
      cov=$6; depth=$7; mapq=$9;
      if (cov>=bmin && depth>=dmin && mapq>=mqmin) print $1
    }' "${COVTSV}" | sort -u > "${KEEP_LIST}"

    if [[ ! -s "${KEEP_LIST}" ]]; then
      echo "[WARN] No contigs for ${sample_name} met the thresholds. Skipping consensus."
      return
    fi
    echo "[*] Kept contigs for ${sample_name}:"
    cat "${KEEP_LIST}"

    local FIL_REF="${sample_outdir}/work/draft.filtered.fa"
    echo "[*] Building filtered reference..."
    seqkit grep -f "${KEEP_LIST}" "${REF}" > "${FIL_REF}"

    echo "[*] Running Medaka consensus..."
    local MEDAKA_DIR="${sample_outdir}/medaka_consensus"
    medaka_consensus -i "${fastq_file}" -d "${FIL_REF}" -o "${MEDAKA_DIR}" -t "${THREADS}" -m "${MODEL}"

    if [[ -f "${MEDAKA_DIR}/consensus.fasta" ]]; then
      local FINAL_FASTA="${sample_outdir}/${sample_name}_consensus.fasta"
      echo "[*] Finalizing consensus FASTA..."
      seqkit grep -f "${KEEP_LIST}" "${MEDAKA_DIR}/consensus.fasta" > "${FINAL_FASTA}"
      echo "[DONE] Pipeline for ${sample_name} finished."
      echo "      Output: ${FINAL_FASTA}"
    else
      echo "[ERR] Medaka did not produce consensus.fasta for ${sample_name}"
    fi
  ) >& "${LOG}" # 함수 내의 모든 출력을 로그 파일로 리디렉션
}


# ==============================================================================
# ===                   메인 스크립트 실행 로직                  ===
# ==============================================================================

# Reference 인덱싱은 처음에 한 번만 수행
if [[ ! -f "${REF}.fai" ]]; then
  echo "[*] Indexing reference FASTA: ${REF}"
  samtools faidx "${REF}"
fi

# --indir 옵션이 주어졌을 경우 (여러 파일 처리)
if [[ -n "${INDIR}" ]]; then
  echo "[INFO] Batch mode started. Processing all FASTQ files in: ${INDIR}"
  mkdir -p "${OUTDIR}"
  
  # 해당 폴더 내의 모든 fastq(.gz) 파일을 순회
  for fq_file in "${INDIR}"/*.{fq,fastq,fq.gz,fastq.gz}; do
    # 파일이 없는 경우를 대비한 안전장치
    [[ -e "$fq_file" ]] || continue
    
    # 각 파일에 대한 개별 결과 폴더 이름 지정
    sample_name=$(basename "${fq_file}" | sed -E 's/\.fq\.gz$|\.fastq\.gz$|\.fq$|\.fastq$//')
    sample_outdir="${OUTDIR}/${sample_name}"
    
    # 파이프라인 함수 호출
    run_pipeline "${fq_file}" "${sample_outdir}"
  done
  echo "[INFO] Batch mode finished. All results are in: ${OUTDIR}"

# --reads 옵션이 주어졌을 경우 (단일 파일 처리)
else
  echo "[INFO] Single file mode started."
  # 단일 파일이므로 메인 outdir을 그대로 사용
  run_pipeline "${READS}" "${OUTDIR}"
fi