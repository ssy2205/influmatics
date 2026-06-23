#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H3N2 HA 분석 도구 (전면 재작성판)
================================================================================

이 스크립트가 하는 일 (요청한 순서대로):
  1) H3N2 HA 서열(target.fasta)을 reference.fasta 에 정렬하고 "H3 넘버링"으로 좌표를 맞춘다.
  2) 코드에 미리 입력된 antigenic site 들의 변이를 확인한다.
  3) antigenic distance 를 계산하고 antigenic cartography(2D 지도)를 그린다.
  4) 같은 H3 넘버링 기준으로, 코드에 미리 입력된 항바이러스제 작용부위의 변이를 확인한다.
  5) clade 를 지정한 뒤, background.fasta 를 포함한 phylogenetic tree 를 그린다.

사용법 (아주 간단):
  - 이 .py 파일과 같은 폴더에 아래 3개 FASTA 를 넣는다.
        target.fasta      : 분석 대상 H3N2 HA 서열(여러 개 가능)
        reference.fasta    : H3 넘버링 기준이 되는 reference HA 서열(1개)
        background.fasta   : 계통수의 배경이 되는 참조 서열들(클레이드 대표주 등)
        vaccine.fasta      : (선택) 항원거리 비교 기준 백신주. 없으면 reference 를 대용으로 사용.
                             antigenic cartography 는 target 과 vaccine(백신주)만 비교한다.
  - 터미널에서:   python h3n2_ha_analysis.py
  - 결과는 results/ 폴더에 표(CSV)와 그림(PNG), 요약(HTML)으로 저장된다.

필요한 패키지 (최초 1회):
        pip install biopython numpy matplotlib

설계 메모:
  - 입력이 핵산(DNA/RNA)이면 단백질로 자동 번역한다. 시퀀싱 raw FASTA 를 그대로 넣어도 된다:
    target/background 는 6프레임(정방향3+역상보3) 중 reference 에 가장 잘 정렬되는 것을 자동 선택하므로
    방향(역상보)·프레임·앞뒤 UTR 을 신경 쓰지 않아도 된다. (reference 는 종결코돈이 가장 적은 프레임 사용)
    각 서열의 reference 일치도(%)를 로그로 출력하며, 40% 미만이면 경고한다.
  - 모든 site 데이터(항원부위/약제부위/클레이드 규칙)는 단백질 H3 넘버링이다.
  - 이 도구는 "관찰된 서열에 알려진 표식을 주석/요약/시각화"하는 후향적 분석 도구다.
    새로운 변이를 설계하거나 적합도/회피를 예측하지 않는다.
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import heapq
import html
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- 외부 패키지 (없으면 친절히 안내하고 종료) -----------------------------------
try:
    import numpy as np
    from Bio import SeqIO, Phylo
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord as BioSeqRecord
    from Bio.Align import PairwiseAligner, MultipleSeqAlignment, substitution_matrices
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    import matplotlib
    matplotlib.use("Agg")  # 화면 없이 파일로만 저장
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # 그래프 안 한글이 깨지지 않도록 한글 폰트 지정(가능한 것 중 첫 번째)
    _available = {f.name for f in font_manager.fontManager.ttflist}
    for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if _font in _available:
            plt.rcParams["font.family"] = _font
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelsize": 10.5,
        "axes.labelcolor": "#3a4149",
        "axes.edgecolor": "#b0b8c0",
        "font.size": 10,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#d0d6dd",
        "legend.fancybox": True,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "[설치 필요] 다음 패키지가 필요합니다:\n"
        "    pip install biopython numpy matplotlib\n"
        f"(원인: {exc})\n"
    )
    raise SystemExit(1)


# ==============================================================================
# ===  ⚙️  설정 영역 — 여기만 고치면 분석 기준이 바뀝니다  ======================
# ==============================================================================

# 입력 파일명(같은 폴더 기준). 필요하면 명령행 인자로 덮어쓸 수 있음.
TARGET_FASTA = "target.fasta"
REFERENCE_FASTA = "reference.fasta"
BACKGROUND_FASTA = "background.fasta"
VACCINE_FASTA = "vaccine.fasta"   # 항원거리 비교 기준이 되는 백신주(없으면 reference 사용)
OUTPUT_DIR = "results"
LOW_IDENTITY_WARN_PERCENT = 40.0
MAX_TREE_SEQUENCES = 0
ANALYSIS_SCOPE_NOTICE = (
    "Retrospective H3N2 HA sequence annotation only. This script compares observed "
    "sequences with curated/reference positions; it does not design, recommend, or "
    "rank new mutations."
)

# --- H3 넘버링 보정값 -----------------------------------------------------------
# H3 넘버링은 "성숙한 HA1"의 첫 잔기를 1번으로 센다.
# reference 서열이 신호펩타이드(signal peptide, H3는 보통 16잔기)를 포함하면 16을 쓴다.
# reference 가 이미 성숙형(HA1 시작)이면 0 으로 바꾼다.
#   reference 의 (1-based) 잔기 번호 = H3site + H3_OFFSET
H3_OFFSET = 16

# --- 항원부위 (H3 넘버링) -------------------------------------------------------
# 사용자가 지정한 기준으로 갱신:
#   Site A: 122, 128, 132-146
#   Site B: 155-160, 186-199
#   Site C: 50, 53-54, 91-92, 275-278
#   Site D: 172-174, 201-207, 217-220, 242-248
#   Site E: 62-63, 78-83
#
# 참고: 이 스크립트는 실험용/후향적 주석 도구이며,
#       항원부위는 H3 넘버링 기준으로만 평가한다.
# 형식: { H3site: (항원부위라벨, 가중치) }
ANTIGENIC_SITES: Dict[int, Tuple[str, float]] = {}
def _add_sites(label: str, positions: List[int], weight: float = 1.0) -> None:
    for p in positions:
        # 이미 있으면 더 큰 가중치를 유지
        prev = ANTIGENIC_SITES.get(p)
        if prev is None or weight > prev[1]:
            ANTIGENIC_SITES[p] = (label, weight)

# 사용자 지정 항원부위 집합
_add_sites("Site_A", [122, 128] + list(range(132, 147)), weight=1.0)
_add_sites("Site_B", list(range(155, 161)) + list(range(186, 200)), weight=1.0)
_add_sites("Site_C", [50] + list(range(53, 55)) + list(range(91, 93)) + list(range(275, 279)), weight=1.0)
_add_sites("Site_D", list(range(172, 175)) + list(range(201, 208)) + list(range(217, 221)) + list(range(242, 249)), weight=1.0)
_add_sites("Site_E", [62, 63] + list(range(78, 84)), weight=1.0)

# 참고용: Koel 2013 핵심 7개 위치는 여전히 가중치 2로 강조
_add_sites("Koel7", [145, 155, 156, 158, 159, 189, 193], weight=2.0)

# --- 항바이러스제 작용부위 (H3 넘버링) -----------------------------------------
# ⚠️ 중요: 고전적 NA 억제제(oseltamivir 등) 내성변이(H275Y 등)는 NA 유전자에 있으며 HA 에는 없다.
#          HA 를 표적하는 약제는 umifenovir(arbidol; HA 줄기/삼량체 계면 결합)가 대표적이다.
#          아래 목록은 "예시/뼈대"이며, 반드시 최신 문헌으로 검증·교체할 것.
#          실제 분석에서 백신주/비교 대상은 별도로 vaccine.fasta 또는 --vaccine 로 설정한다.
# 형식: { H3site: (약제, 설명) }
DRUG_SITES: Dict[int, Tuple[str, str]] = {
    # arbidol(umifenovir) 결합 포켓/삼량체 계면 근처로 보고된 예시 위치 (검증 필요)
    291: ("umifenovir(arbidol)", "예시: HA 삼량체 계면 인접 — 출처 검증 필요"),
    312: ("umifenovir(arbidol)", "예시: HA 삼량체 계면 인접 — 출처 검증 필요"),
    # HA2 fusion peptide / 줄기 영역 예시 (HA2 좌표는 별도 — 아래 note 참조)
    # 필요시 사용자가 본인 분석의 검증된 위치로 교체.
}
DRUG_SITES_NOTE = (
    "DRUG_SITES 는 예시 뼈대입니다. NA 억제제 내성변이는 HA 가 아니라 NA 에 있으므로, "
    "HA 입력으로는 평가할 수 없습니다. HA 표적 약제(arbidol 등)의 위치를 최신 문헌으로 채우세요."
)

# --- 클레이드 정의 규칙 (H3 넘버링) ---------------------------------------------
# ⚠️ 예시 규칙입니다. 실제 분석에서는 Nextstrain/Nextclade 의 현행 정의로 교체하세요.
# 형식: { 클레이드명: { H3site: 기대아미노산 } }  (규칙 충족 비율이 가장 높은 클레이드로 지정)
CLADE_RULES: Dict[str, Dict[int, str]] = {
    "3C.2a1b.2a.2 (예시)": {159: "Y", 193: "S", 131: "K", 142: "K"},
    "3C.2a1b.2a.1 (예시)": {159: "Y", 193: "F", 50: "E"},
    "3C.2a1b.1a (예시)": {159: "N", 131: "K", 142: "G"},
}
CLADE_MIN_SCORE = 0.6  # 이 비율 이상 규칙이 맞아야 클레이드를 지정

# 실제 클레이드 호출은 외부 CLI(예: nextclade)가 있으면 우선 사용한다.
# 있으면 해당 결과를 쓰고, 없으면 아래 규칙 기반으로 떨어진다.

# 클레이드 색상 팔레트(트리 색칠용)
PALETTE = ["#2E86AB", "#F18F01", "#C73E1D", "#6A4C93", "#1B998B",
           "#B56576", "#5C677D", "#4CAF50", "#D95D39", "#3B8EA5"]


# ==============================================================================
# ===  1. 입출력 & 서열 준비  ===================================================
# ==============================================================================

NT_ALPHABET = set("ACGTUN-")


try:  # Windows 콘솔에서 한글이 깨지지 않도록
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def log(msg: str) -> None:
    print(f"[h3n2-ha] {msg}", flush=True)


def write_fasta(path: Path, records: Iterable[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{seq}\n")


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    """FASTA 를 [(id, sequence), ...] 로 읽는다."""
    records: List[Tuple[str, str]] = []
    for rec in SeqIO.parse(str(path), "fasta"):
        records.append((rec.id, str(rec.seq).upper().replace(" ", "")))
    if not records:
        raise ValueError(f"FASTA 에 서열이 없습니다: {path}")
    return records


def is_nucleotide(seq: str) -> bool:
    letters = [c for c in seq.upper() if not c.isspace()]
    if not letters:
        return False
    nt = sum(1 for c in letters if c in NT_ALPHABET)
    return nt / len(letters) >= 0.9


def _clean_nt(seq: str) -> str:
    return re.sub(r"[^ACGTN]", "", seq.upper().replace("U", "T"))


def six_frame_proteins(seq: str) -> List[str]:
    """염기서열의 6프레임(정방향 3 + 역상보 3) 번역 후보를 모두 반환."""
    s = _clean_nt(seq)
    cands: List[str] = []
    for strand in (s, str(Seq(s).reverse_complement())):
        for frame in range(3):
            sub = strand[frame:]
            sub = sub[: len(sub) // 3 * 3]
            if sub:
                cands.append(str(Seq(sub).translate()))
    return cands


def to_protein(seq: str) -> str:
    """[reference 용] 핵산이면 6프레임 중 내부 종결코돈이 가장 적은 것으로 번역. 단백질이면 그대로.
    reference 는 비교 대상이 없으므로 종결코돈 기준으로 가장 깔끔한 프레임을 고른다."""
    s = re.sub(r"[^A-Z*]", "", seq.upper())
    if not is_nucleotide(s):
        return s  # 이미 단백질
    best: Optional[Tuple[int, int, str]] = None  # (내부종결수, -길이, 단백질)
    for prot in six_frame_proteins(s):
        internal_stops = prot.rstrip("*").count("*")
        key = (internal_stops, -len(prot.rstrip("*")))
        if best is None or key < (best[0], best[1]):
            best = (internal_stops, -len(prot.rstrip("*")), prot)
    return (best[2] if best else "").rstrip("*")


def to_protein_vs_reference(seq: str, ref_prot: str, aligner: "PairwiseAligner") -> str:
    """[target/background 용] 핵산이면 6프레임 중 reference 에 가장 잘 정렬되는 것을 선택.
    정방향/역상보·프레임을 자동 결정하고, UTR 등 양끝 잡음은 정렬 단계에서 잘린다.
    단백질이면 그대로 반환."""
    s = re.sub(r"[^A-Z*]", "", seq.upper())
    if not is_nucleotide(s):
        return s  # 이미 단백질
    best: Optional[Tuple[float, str]] = None
    for prot in six_frame_proteins(s):
        stripped = prot.strip("*")
        if not stripped:
            continue
        try:
            score = float(aligner.score(ref_prot, stripped))
        except Exception:
            continue
        if best is None or score > best[0]:
            best = (score, stripped)
    return best[1] if best else ""


def projection_identity(ref_proj: str, proj: str) -> float:
    """reference 대비 일치도(%) — 양쪽 모두 잔기가 있는 위치만 비교."""
    compared = matched = 0
    for a, b in zip(ref_proj, proj):
        if a in "-X" or b in "-X":
            continue
        compared += 1
        if a == b:
            matched += 1
    return round(100.0 * matched / compared, 1) if compared else 0.0


def normalize_id(name: str) -> str:
    name = re.sub(r"[\s/|:;,()\[\]']+", "_", str(name).strip())
    return re.sub(r"_+", "_", name).strip("_") or "seq"


def tree_label_key(name: str) -> str:
    """R ggtree script style label key for matching tip names and outlier lists."""
    text = re.sub(r"[\"'()\[\]]", "", str(name).strip())
    text = re.sub(r"[\s/|:;,\.\-]+", "_", text)
    text = re.sub(r"(__[A-Za-z0-9]+_)+$", "", text)
    return re.sub(r"_+", "_", text).strip("_")


def tree_name_keys(name: str) -> List[str]:
    """Return robust lowercase keys for matching FASTA ids to metadata tables."""
    keys = {
        str(name).strip(),
        normalize_id(name),
        tree_label_key(name),
    }
    return [key.lower() for key in keys if key]


def set_date_override(mapping: Dict[str, str], name: str, date_value: str) -> None:
    value = str(date_value).strip()
    if not value:
        return
    for key in tree_name_keys(name):
        mapping[key] = value


def find_date_override(mapping: Optional[Dict[str, str]], name: str) -> str:
    if not mapping:
        return ""
    for key in tree_name_keys(name):
        value = mapping.get(key)
        if value:
            return value
    return ""


def extract_collection_date(name: str) -> str:
    """Extract YYYY-MM-DD, YYYY-MM, or YYYY from common influenza FASTA ids."""
    text = str(name)
    full_matches = re.findall(
        r"(?<!\d)((?:19|20)\d{2})[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])(?!\d)",
        text,
    )
    if full_matches:
        y, m, d = full_matches[-1]
        return f"{y}-{m}-{d}"
    month_matches = re.findall(r"(?<!\d)((?:19|20)\d{2})[-_](0[1-9]|1[0-2])(?!\d)", text)
    if month_matches:
        y, m = month_matches[-1]
        return f"{y}-{m}"
    year_matches = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    if year_matches:
        return year_matches[-1]
    return ""


def collection_date_to_decimal_year(date_value: str) -> Optional[float]:
    """Convert YYYY, YYYY-MM, or YYYY-MM-DD to a decimal year for plotting."""
    if not date_value:
        return None
    match = re.match(r"^((?:19|20)\d{2})(?:-(\d{2})(?:-(\d{2}))?)?$", date_value)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2) or "7")
    day = int(match.group(3) or "15")
    month_lengths = [31, 28 + int((year % 4 == 0 and year % 100 != 0) or year % 400 == 0),
                     31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = min(max(month, 1), 12)
    day = min(max(day, 1), month_lengths[month - 1])
    elapsed = sum(month_lengths[:month - 1]) + day - 1
    return year + elapsed / sum(month_lengths)


def extract_strain_year(name: str) -> Optional[int]:
    """Extract the isolate/strain year from the name before metadata suffixes."""
    prefix = re.split(r"(?:\|?EPI_ISL_|_EPI_ISL_)", str(name), maxsplit=1)[0]
    prefix = re.sub(
        r"[-_/]?(?:19|20)\d{2}[-_](?:0[1-9]|1[0-2])[-_](?:0[1-9]|[12]\d|3[01])$",
        "",
        prefix,
    )
    years = [int(year) for year in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", prefix)]
    if not years:
        return None
    return years[-1]


def find_temporal_metadata_conflicts(
    names: Iterable[str],
    protected_names: Iterable[str],
    date_overrides: Optional[Dict[str, str]] = None,
    mismatch_years: int = 3,
) -> List[Dict[str, object]]:
    """Find tips whose collection date strongly conflicts with the strain year."""
    overrides = date_overrides or {}
    protected_keys = {tree_label_key(name).lower() for name in protected_names}
    protected_keys.update(normalize_id(name).lower() for name in protected_names)
    rows: List[Dict[str, object]] = []
    for name in names:
        name_keys = {tree_label_key(name).lower(), normalize_id(name).lower()}
        if name_keys & protected_keys:
            continue
        date_value = find_date_override(overrides, name) or extract_collection_date(name)
        if not date_value:
            continue
        match = re.match(r"^((?:19|20)\d{2})", date_value)
        if not match:
            continue
        collection_year = int(match.group(1))
        strain_year = extract_strain_year(name)
        if strain_year is None:
            continue
        diff = abs(collection_year - strain_year)
        if diff > mismatch_years:
            rows.append({
                "sample": name,
                "collection_date": date_value,
                "collection_year": collection_year,
                "strain_year": strain_year,
                "year_difference": diff,
                "reason": "collection_date_conflicts_with_strain_year",
            })
    return rows


def six_frame_coding_nt(seq: str) -> List[Tuple[str, List[str]]]:
    """Return translated protein and codons for all six nucleotide frames."""
    s = _clean_nt(seq)
    candidates: List[Tuple[str, List[str]]] = []
    for strand in (s, str(Seq(s).reverse_complement())):
        for frame in range(3):
            sub = strand[frame:]
            sub = sub[: len(sub) // 3 * 3]
            if not sub:
                continue
            codons = [sub[i:i + 3] for i in range(0, len(sub), 3)]
            prot = str(Seq(sub).translate())
            while prot.endswith("*") and codons:
                prot = prot[:-1]
                codons = codons[:-1]
            candidates.append((prot, codons))
    return candidates


def project_coding_nt_to_reference(
    raw_seq: str,
    ref_prot: str,
    aligner: "PairwiseAligner",
) -> str:
    """Project a nucleotide coding sequence to the reference protein coordinates.

    The output is a codon alignment with length len(ref_prot) * 3. Insertions
    relative to the reference protein are intentionally dropped to keep stable
    HA coordinates for IQ-TREE/TreeTime.
    """
    s = re.sub(r"[^A-Z*]", "", raw_seq.upper())
    if not is_nucleotide(s):
        return ""

    best: Optional[Tuple[float, str, List[str]]] = None
    for prot, codons in six_frame_coding_nt(s):
        if not prot or not codons:
            continue
        try:
            score = float(aligner.score(ref_prot, prot))
        except Exception:
            continue
        if best is None or score > best[0]:
            best = (score, prot, codons)
    if best is None:
        return ""

    _, prot, codons = best
    aln = aligner.align(ref_prot, prot)[0]
    ref_blocks, qry_blocks = aln.aligned
    projected = ["---"] * len(ref_prot)
    for (rs, re_), (qs, qe) in zip(ref_blocks, qry_blocks):
        for off in range(min(re_ - rs, qe - qs)):
            qidx = qs + off
            if 0 <= qidx < len(codons):
                codon = codons[qidx]
                projected[rs + off] = codon if len(codon) == 3 else "---"
    return "".join(projected)


def pick_column(rows: List[Dict[str, str]], candidates: List[str]) -> Optional[str]:
    if not rows:
        return None
    lower_to_real = {key.lower(): key for key in rows[0].keys()}
    for candidate in candidates:
        real = lower_to_real.get(candidate.lower())
        if real is not None:
            return real
    return None


def read_delimited_table(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        if path.suffix.lower() not in {".csv", ".tsv", ".tab"}:
            try:
                delimiter = csv.Sniffer().sniff(sample).delimiter
            except csv.Error:
                pass
        dialect = csv.excel_tab if delimiter == "\t" else csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in reader]


def load_tree_date_metadata(path: Path) -> Tuple[Dict[str, str], int]:
    """Load a name/date metadata table for TreeTime tip dating."""
    if not path.exists():
        raise FileNotFoundError(f"TreeTime date metadata file not found: {path}")
    rows = read_delimited_table(path)
    name_col = pick_column(rows, ["name", "seqName", "seq_name", "strain", "id", "sample", "sample_id", "tip"])
    date_col = pick_column(
        rows,
        [
            "date",
            "collection_date",
            "collectionDate",
            "date_collected",
            "collected",
            "year",
            "decimal_date",
        ],
    )
    if not name_col or not date_col:
        raise ValueError(
            "TreeTime date metadata must contain name/id and date columns "
            "(for example: name,date)."
        )
    metadata: Dict[str, str] = {}
    used_rows = 0
    for row in rows:
        name = row.get(name_col, "").strip()
        date_value = row.get(date_col, "").strip()
        if not name or not date_value:
            continue
        set_date_override(metadata, name, date_value)
        used_rows += 1
    return metadata, used_rows


def _json_get(row: Dict[str, Any], candidates: List[str]) -> str:
    for key in candidates:
        cur: Any = row
        ok = True
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur not in (None, ""):
            return str(cur)
    return ""


def load_nextclade_clade_file(path: Path) -> Dict[str, Dict[str, str]]:
    """Load Nextclade CSV/TSV/JSON and return normalized_id -> clade metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Nextclade result file not found: {path}")

    seq_candidates = ["seqName", "seq_name", "name", "strain", "id", "sample", "sample_id"]
    clade_candidates = [
        "clade", "nextstrainClade", "nextstrain_clade", "clade_nextstrain",
        "legacy.clade", "Nextclade_pango", "nextclade.clade", "pangoLineage",
    ]
    qc_candidates = ["qc.overallStatus", "qc_status", "qc.overallScore", "qc_status_overall"]
    error_candidates = ["errors", "error", "warnings", "warning"]

    out: Dict[str, Dict[str, str]] = {}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            rows = data.get("results") or data.get("data") or data.get("rows") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        if not isinstance(rows, list):
            raise ValueError(f"Unsupported Nextclade JSON structure: {path}")
        for row in rows:
            if not isinstance(row, dict):
                continue
            nested = row.get("analysisResult") if isinstance(row.get("analysisResult"), dict) else {}
            merged = {**row, **nested}
            seq_name = _json_get(merged, seq_candidates)
            clade = _json_get(merged, clade_candidates)
            qc_status = _json_get(merged, qc_candidates)
            errors = _json_get(merged, error_candidates)
            norm = normalize_id(seq_name)
            if norm:
                out[norm] = {
                    "clade": clade or "unassigned",
                    "qc_status": qc_status,
                    "errors": errors,
                }
        return out

    rows = read_delimited_table(path)
    seq_col = pick_column(rows, seq_candidates)
    clade_col = pick_column(rows, clade_candidates)
    qc_col = pick_column(rows, qc_candidates)
    error_col = pick_column(rows, error_candidates)
    if seq_col is None:
        raise ValueError("Nextclade CSV/TSV needs a seqName/name/id column.")
    if clade_col is None:
        raise ValueError("Nextclade CSV/TSV needs a clade column.")
    for row in rows:
        seq_name = row.get(seq_col, "")
        clade = row.get(clade_col, "")
        norm = normalize_id(seq_name)
        if norm:
            out[norm] = {
                "clade": clade or "unassigned",
                "qc_status": row.get(qc_col, "") if qc_col else "",
                "errors": row.get(error_col, "") if error_col else "",
            }
    return out


def resolve_input_path(base: Path, value: str) -> Path:
    """Resolve CLI file paths relative to the script directory unless already absolute."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def find_nextclade_executable(base: Path) -> Optional[str]:
    found = shutil.which("nextclade") or shutil.which("nextclade.exe")
    if found:
        return found
    for candidate in [
        Path.cwd() / ".tools" / "nextclade.exe",
        base / ".tools" / "nextclade.exe",
        base.parent / ".tools" / "nextclade.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def evenly_sample_items(items: List[Tuple[str, str]], max_items: int) -> Dict[str, str]:
    """Return a deterministic, evenly spaced subset without random sampling."""
    if max_items <= 0 or len(items) <= max_items:
        return dict(items)
    if max_items == 1:
        return dict([items[0]])
    step = (len(items) - 1) / (max_items - 1)
    picked = []
    seen = set()
    for i in range(max_items):
        idx = round(i * step)
        name, seq = items[idx]
        if name not in seen:
            picked.append((name, seq))
            seen.add(name)
    return dict(picked)


def build_tree_input(
    ref_id: str,
    ref_proj: str,
    targets: Dict[str, str],
    background: Dict[str, str],
    vaccines: Dict[str, str],
    max_sequences: int,
) -> Tuple[Dict[str, str], int]:
    """Keep reference/targets/vaccines and downsample background only when needed."""
    tree_input: Dict[str, str] = {normalize_id(ref_id): ref_proj}
    tree_input.update(targets)
    tree_input.update(vaccines)
    reserved = len(tree_input)
    if max_sequences <= 0:
        tree_input.update(background)
        return tree_input, 0
    remaining = max(max_sequences - reserved, 0)
    selected_background = evenly_sample_items(list(background.items()), remaining)
    tree_input.update(selected_background)
    skipped = max(0, len(background) - len(selected_background))
    return tree_input, skipped


def split_tree_outlier_terms(text: str) -> List[str]:
    terms: List[str] = []
    for line in str(text or "").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in re.split(r"[,;\t]+", line):
            item = part.strip().strip("\"'")
            if item:
                terms.append(item)
    return terms


def load_tree_outlier_terms(
    inline_terms: str,
    file_path: Optional[str],
    base: Path,
) -> List[str]:
    terms = split_tree_outlier_terms(inline_terms)
    if file_path:
        path = resolve_input_path(base, file_path)
        if not path.exists():
            raise FileNotFoundError(f"tree outlier file not found: {path}")
        terms.extend(split_tree_outlier_terms(path.read_text(encoding="utf-8-sig")))

    seen = set()
    unique_terms: List[str] = []
    for term in terms:
        key = tree_label_key(term).lower()
        if key and key not in seen:
            seen.add(key)
            unique_terms.append(term)
    return unique_terms


def _outlier_term_matches_name(term: str, name: str) -> bool:
    term_keys = {str(term).strip(), normalize_id(term), tree_label_key(term)}
    name_keys = {str(name).strip(), normalize_id(name), tree_label_key(name)}
    term_keys = {k.lower() for k in term_keys if k}
    name_keys = {k.lower() for k in name_keys if k}
    if term_keys & name_keys:
        return True
    for term_key in term_keys:
        if len(term_key) >= 8 and any(term_key in name_key for name_key in name_keys):
            return True
    return False


def _collection_year(name: str) -> Optional[int]:
    date_value = extract_collection_date(name)
    if not date_value:
        return None
    match = re.match(r"^((?:19|20)\d{2})", date_value)
    if not match:
        return None
    return int(match.group(1))


def filter_tree_outliers(
    tree_input: Dict[str, str],
    clade_by_name: Dict[str, str],
    outlier_terms: List[str],
    protected_names: Iterable[str],
    out_csv: Path,
    date_min: Optional[int] = None,
    date_max: Optional[int] = None,
) -> Tuple[Dict[str, str], List[Dict[str, object]]]:
    protected_keys = {tree_label_key(name).lower() for name in protected_names}
    protected_keys.update(normalize_id(name).lower() for name in protected_names)
    removed_rows: List[Dict[str, object]] = []
    filtered: Dict[str, str] = {}

    for name, seq in tree_input.items():
        name_keys = {tree_label_key(name).lower(), normalize_id(name).lower()}
        if name_keys & protected_keys:
            filtered[name] = seq
            continue

        reason = ""
        matched_query = ""
        for term in outlier_terms:
            if _outlier_term_matches_name(term, name):
                reason = "manual_list"
                matched_query = term
                break

        year = _collection_year(name)
        if not reason and year is not None:
            if date_min is not None and year < date_min:
                reason = f"date_before_{date_min}"
            elif date_max is not None and year > date_max:
                reason = f"date_after_{date_max}"

        if reason:
            removed_rows.append({
                "sample": name,
                "assigned_clade": clade_by_name.get(name, "unassigned"),
                "reason": reason,
                "matched_query": matched_query,
                "collection_date": extract_collection_date(name),
            })
        else:
            filtered[name] = seq

    write_csv(out_csv, removed_rows,
              ["sample", "assigned_clade", "reason", "matched_query", "collection_date"])
    return filtered, removed_rows


# ==============================================================================
# ===  2. reference 정렬 & H3 넘버링 좌표 매핑  =================================
# ==============================================================================

def make_aligner() -> "PairwiseAligner":
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    # 말단 갭은 무료(서열 길이가 달라도 잘 붙도록). biopython 버전차 대비 try.
    try:
        aligner.end_gap_score = 0.0
    except Exception:
        pass
    return aligner


def align_to_reference(ref_seq: str, query_seq: str, aligner: "PairwiseAligner") -> str:
    """query 를 reference 좌표로 투영한다.
    반환 길이는 len(ref_seq) 이며, reference 각 위치에 대응하는 query 잔기(없으면 '-')를 담는다.
    (reference 기준 삽입은 버린다 — H3 넘버링 좌표를 유지하기 위함)
    """
    if not query_seq:
        return "-" * len(ref_seq)
    aln = aligner.align(ref_seq, query_seq)[0]
    ref_blocks, qry_blocks = aln.aligned  # 각: [(start,end), ...] (0-based)
    projected = ["-"] * len(ref_seq)
    for (rs, re_), (qs, qe) in zip(ref_blocks, qry_blocks):
        for off in range(re_ - rs):
            projected[rs + off] = query_seq[qs + off]
    return "".join(projected)


def h3site_to_index(site: int) -> int:
    """H3 넘버링 site -> reference-투영 서열의 0-based 인덱스."""
    return site + H3_OFFSET - 1


def residue_at(projected_seq: str, site: int) -> str:
    idx = h3site_to_index(site)
    if 0 <= idx < len(projected_seq):
        return projected_seq[idx]
    return ""  # 좌표 밖(커버리지 부족)


# ==============================================================================
# ===  3. 항원부위 변이 확인  ===================================================
# ==============================================================================

def antigenic_mutation_table(
    ref_proj: str,
    samples: Dict[str, str],
) -> List[Dict[str, object]]:
    """각 샘플에서 항원부위별 변이(reference 대비)를 표로 만든다."""
    rows: List[Dict[str, object]] = []
    for name, proj in samples.items():
        for site in sorted(ANTIGENIC_SITES):
            label, weight = ANTIGENIC_SITES[site]
            ref_aa = residue_at(ref_proj, site)
            obs_aa = residue_at(proj, site)
            if not ref_aa or not obs_aa or obs_aa in "-X":
                continue
            if obs_aa != ref_aa:
                rows.append({
                    "sample": name,
                    "antigenic_site": label,
                    "h3_position": site,
                    "reference_aa": ref_aa,
                    "observed_aa": obs_aa,
                    "mutation": f"{ref_aa}{site}{obs_aa}",
                    "weight": weight,
                })
    return rows


# ==============================================================================
# ===  4. Antigenic distance & cartography  ====================================
# ==============================================================================

def antigenic_distance(proj_a: str, proj_b: str) -> float:
    """항원부위에서의 가중 해밍거리(0~1)."""
    total_w = 0.0
    diff_w = 0.0
    for site, (_, weight) in ANTIGENIC_SITES.items():
        a = residue_at(proj_a, site)
        b = residue_at(proj_b, site)
        if not a or not b or a in "-X" or b in "-X":
            continue
        total_w += weight
        if a != b:
            diff_w += weight
    return (diff_w / total_w) if total_w else 0.0


def antigenic_distance_to_vaccine(
    targets: Dict[str, str],
    vaccines: Dict[str, str],
) -> List[Dict[str, object]]:
    """각 target 이 각 백신주에서 항원적으로 얼마나 떨어졌는지 + 어느 항원부위가 다른지."""
    rows: List[Dict[str, object]] = []
    for tname, tproj in targets.items():
        for vname, vproj in vaccines.items():
            total_w = diff_w = 0.0
            compared = diffs = 0
            differing: List[str] = []
            for site, (_, weight) in sorted(ANTIGENIC_SITES.items()):
                a = residue_at(tproj, site)
                b = residue_at(vproj, site)
                if not a or not b or a in "-X" or b in "-X":
                    continue
                total_w += weight
                compared += 1
                if a != b:
                    diff_w += weight
                    diffs += 1
                    differing.append(f"{b}{site}{a}")  # 백신주잔기 + 위치 + 샘플잔기
            dist = (diff_w / total_w) if total_w else 0.0
            rows.append({
                "sample": tname,
                "vaccine": vname,
                "antigenic_distance": round(dist, 4),
                "sites_compared": compared,
                "antigenic_differences": diffs,
                "differing_sites": ";".join(differing),
            })
    return rows


def classical_mds(dmat: List[List[float]]) -> List[Tuple[float, float]]:
    """고전적 MDS(주좌표분석)로 거리행렬을 2D 좌표로 변환."""
    n = len(dmat)
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]
    d = np.array(dmat, dtype=float)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j.dot(d ** 2).dot(j)
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    coords = np.zeros((n, 2))
    for axis in range(min(2, n)):
        if vals[axis] > 0:
            coords[:, axis] = vecs[:, axis] * math.sqrt(float(vals[axis]))
    return [(float(x), float(y)) for x, y in coords]


def clade_color_map(clades: Iterable[str]) -> Dict[str, str]:
    """클레이드 -> 색. 두 그림(트리·cartography)에서 색을 통일하기 위한 공용 함수."""
    real = sorted(c for c in set(clades) if c and c != "unassigned")
    cmap = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(real)}
    cmap["unassigned"] = "#9aa5b1"
    cmap[""] = "#9aa5b1"
    return cmap


def _style_axes(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors="#6b7480", labelsize=9)
    ax.grid(True, color="#e9edf2", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)


def draw_cartography(
    names: List[str],
    groups: List[str],
    coords: List[Tuple[float, float]],
    clade_by_name: Dict[str, str],
    out_png: Path,
) -> None:
    clades = sorted(set(clade_by_name.get(n, "") for n in names))
    cmap = clade_color_map(clades)
    marker_for = {"vaccine": "D", "reference": "D", "target": "*", "background": "o"}
    size_for = {"vaccine": 150, "reference": 110, "target": 300, "background": 80}
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    for (x, y), name, grp in zip(coords, names, groups):
        col = cmap.get(clade_by_name.get(name, ""), "#5c677d")
        ax.scatter(x, y, marker=marker_for.get(grp, "o"), s=size_for.get(grp, 80),
                   facecolor=col, edgecolors="#2b2f36", linewidths=0.8,
                   alpha=0.92, zorder=3)
        ax.annotate(name, (x, y), fontsize=7, xytext=(6, 4),
                    textcoords="offset points", color="#3a4149")
    _style_axes(ax)
    ax.set_title("Antigenic cartography (sequence-based)")
    ax.set_xlabel("antigenic dimension 1")
    ax.set_ylabel("antigenic dimension 2")
    # 범례 2개: 색=clade, 모양=group
    clade_handles = [plt.Line2D([0], [0], marker="o", linestyle="none",
                                markerfacecolor=cmap[c], markeredgecolor="white",
                                markersize=10, label=c) for c in clades]
    leg1 = ax.legend(handles=clade_handles, loc="upper left", fontsize=8,
                     title="clade", title_fontsize=9)
    ax.add_artist(leg1)
    present_groups = [g for g in ("target", "vaccine", "reference", "background")
                      if g in set(groups)]
    grp_handles = [plt.Line2D([0], [0], marker=marker_for.get(g, "o"), linestyle="none",
                              markerfacecolor="#8a929b", markeredgecolor="#2b2f36",
                              markersize=11, label=g)
                   for g in present_groups]
    ax.legend(handles=grp_handles, loc="lower right", fontsize=8,
              title="group", title_fontsize=9)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ==============================================================================
# ===  5. 약제 작용부위 변이 확인  ==============================================
# ==============================================================================

def drug_mutation_table(ref_proj: str, samples: Dict[str, str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, proj in samples.items():
        for site in sorted(DRUG_SITES):
            drug, note = DRUG_SITES[site]
            ref_aa = residue_at(ref_proj, site)
            obs_aa = residue_at(proj, site)
            if not ref_aa or not obs_aa or obs_aa in "-X":
                continue
            rows.append({
                "sample": name,
                "drug": drug,
                "h3_position": site,
                "reference_aa": ref_aa,
                "observed_aa": obs_aa,
                "mutation": f"{ref_aa}{site}{obs_aa}",
                "changed": obs_aa != ref_aa,
                "note": note,
            })
    return rows


# ==============================================================================
# ===  6. 클레이드 지정  ========================================================
# ==============================================================================

def assign_clade(proj: str) -> Tuple[str, float]:
    """규칙 충족 비율이 가장 높은 클레이드를 반환. (clade명, 점수)"""
    best_name, best_score = "unassigned", 0.0
    for clade, rules in CLADE_RULES.items():
        tested = matched = 0
        for site, expected in rules.items():
            obs = residue_at(proj, site)
            if not obs or obs in "-X":
                continue
            tested += 1
            if obs == expected:
                matched += 1
        score = (matched / tested) if tested else 0.0
        if score > best_score:
            best_name, best_score = clade, score
    if best_score < CLADE_MIN_SCORE:
        return "unassigned", round(best_score, 3)
    return best_name, round(best_score, 3)


def run_nextclade_clade_call(
    query_records: Iterable[Tuple[str, str]],
    ref_path: Path,
    out_json: Path,
    dataset: Optional[str] = None,
    cli_path: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Nextclade CLI가 있으면 실제 클레이드 호출을 수행한다.

    query_records: FASTA 형태의 서열들[(name, seq)]
    ref_path: reference FASTA 경로
    out_json: Nextclade JSON 결과 저장 경로
    dataset: optional dataset path/name
    """
    cli = cli_path or shutil.which("nextclade") or shutil.which("nextclade.exe")
    if not cli:
        raise FileNotFoundError("nextclade not found")

    query_file = out_json.parent / "nextclade_queries.fasta"
    write_fasta(query_file, query_records)

    out_tsv = out_json.with_suffix(".tsv")
    cmd = [
        cli, "run",
        "--input-ref", str(ref_path),
        "--output-json", str(out_json),
        "--output-tsv", str(out_tsv),
        str(query_file),
    ]
    if dataset:
        cmd = [
            cli, "run",
            "--input-dataset", dataset,
            "--output-json", str(out_json),
            "--output-tsv", str(out_tsv),
            str(query_file),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Nextclade 실행 실패 (exit={result.returncode})\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    if out_tsv.exists():
        return load_nextclade_clade_file(out_tsv)
    return load_nextclade_clade_file(out_json)


# ==============================================================================
# ===  7. 계통수(NJ) + 클레이드 색칠  ==========================================
# ==============================================================================

def build_tree_png(
    aligned: Dict[str, str],
    clade_by_name: Dict[str, str],
    out_png: Path,
    out_newick: Path,
    target_names: Optional[Iterable[str]] = None,
    display_note: str = "",
) -> None:
    # reference 좌표로 투영된 동일 길이 서열들로 MSA 구성
    msa = MultipleSeqAlignment(
        [BioSeqRecord(Seq(seq), id=name, description="") for name, seq in aligned.items()]
    )
    dm = DistanceCalculator("identity").get_distance(msa)
    tree = DistanceTreeConstructor().nj(dm)
    tree.ladderize()
    Phylo.write(tree, str(out_newick), "newick")

    clades = sorted(set(clade_by_name.values()))
    cmap = clade_color_map(clades)

    def color_for(name: str) -> str:
        return cmap.get(clade_by_name.get(name, ""), "#5c677d")

    targets = set(target_names or [])

    # --- 좌표 계산 (음수/0 가지길이 대비해 직접 깊이 계산) -----------------------
    depth: Dict[object, float] = {}
    def comp_depth(clade, acc: float) -> None:
        bl = clade.branch_length or 0.0
        if bl < 0:
            bl = 0.0
        depth[clade] = acc + bl
        for child in clade.clades:
            comp_depth(child, depth[clade])
    comp_depth(tree.root, 0.0)

    terminals = tree.get_terminals()
    n = len(terminals)
    ypos: Dict[object, float] = {t: i for i, t in enumerate(terminals)}
    def assign_y(clade) -> float:
        if clade.is_terminal():
            return ypos[clade]
        ys = [assign_y(c) for c in clade.clades]
        ypos[clade] = sum(ys) / len(ys)
        return ypos[clade]
    assign_y(tree.root)

    xmax = max(depth.values()) or 1.0

    # --- 그리기 ---------------------------------------------------------------
    fig_h = max(7.0, min(14.5, n * 0.0095))
    fig, ax = plt.subplots(figsize=(18, fig_h), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    for clade in tree.find_clades():
        x, y = depth[clade], ypos[clade]
        for child in clade.clades:
            cx, cy = depth[child], ypos[child]
            edge_col = color_for(child.name) if child.is_terminal() else "#9ea8b6"
            ax.plot([x, x], [y, cy], color="#d7dde5", lw=0.8,
                    solid_capstyle="round", alpha=0.9, zorder=1)
            ax.plot([x, cx], [cy, cy], color=edge_col, lw=1.2,
                    solid_capstyle="round", alpha=0.92, zorder=2)

    for t in terminals:
        x, y = depth[t], ypos[t]
        col = color_for(t.name)
        if t.name in targets:
            ax.scatter([x], [y], s=170, marker="*",
                       facecolor="#ffd166", edgecolors="#111827",
                       linewidths=0.9, zorder=6, alpha=1.0)
            label = f"TARGET: {t.name}"
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(12, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "#fff7d6",
                    "edgecolor": "#111827",
                    "linewidth": 0.8,
                    "alpha": 0.96,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#111827",
                    "linewidth": 0.8,
                    "shrinkA": 0,
                    "shrinkB": 4,
                },
                zorder=7,
                clip_on=False,
            )
        else:
            ax.scatter([x], [y], s=10, facecolor=col, edgecolors="#ffffff",
                       linewidths=0.35, zorder=3, alpha=0.95)

    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(-xmax * 0.03, xmax * 1.36)
    ax.set_yticks([])
    ax.margins(x=0.02, y=0.02)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cfd7de")
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="x", colors="#5c6773", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("evolutionary distance (substitutions per site)", fontsize=10)
    title = "Phylogenetic tree  ·  Neighbor-Joining"
    if display_note:
        title = f"{title}\n{display_note}"
    ax.set_title(title, fontsize=12, pad=8)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="none",
                          markerfacecolor=cmap[c], markeredgecolor="#f8f9fb",
                          markersize=8, label=c) for c in clades]
    if targets:
        handles.append(
            plt.Line2D(
                [0], [0], marker="*", linestyle="none",
                markerfacecolor="#ffd166", markeredgecolor="#111827",
                markersize=12, label="target sequence",
            )
        )
    ax.legend(handles=handles, loc="lower left", fontsize=9,
              title="clade", title_fontsize=10,
              frameon=True, fancybox=False, framealpha=0.92,
              borderpad=0.45, labelspacing=0.5,
              handletextpad=0.5)

    plt.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.08)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


class FastTreeNode:
    """Small tree node for fast full-background UPGMA rendering."""

    __slots__ = ("id", "name", "left", "right", "height", "size", "x", "y")

    def __init__(
        self,
        node_id: int,
        name: Optional[str] = None,
        left: Optional["FastTreeNode"] = None,
        right: Optional["FastTreeNode"] = None,
        height: float = 0.0,
        size: int = 1,
    ) -> None:
        self.id = node_id
        self.name = name
        self.left = left
        self.right = right
        self.height = height
        self.size = size
        self.x = 0.0
        self.y = 0.0

    def is_leaf(self) -> bool:
        return self.name is not None

    def children(self) -> List["FastTreeNode"]:
        return [c for c in (self.left, self.right) if c is not None]


def _newick_label(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.|:/-]+", "_", name).strip("_")
    return cleaned or "unnamed"


def _fast_upgma_root(aligned: Dict[str, str]) -> FastTreeNode:
    """Average-linkage/UPGMA tree from equal-length projected sequences.

    This is intended for dense dashboard-scale visualization. It avoids the
    cubic Neighbor-Joining bottleneck for ~1k+ sequence full-background trees.
    """
    names = list(aligned.keys())
    seqs = [aligned[n] for n in names]
    n = len(seqs)
    if n == 0:
        raise ValueError("No sequences for tree")
    if n == 1:
        return FastTreeNode(0, name=names[0])

    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError("Fast UPGMA requires equal-length projected sequences")
    seq_len = lengths.pop()
    if seq_len == 0:
        raise ValueError("Empty projected sequences")

    joined = "".join(seqs).encode("ascii", errors="replace")
    arr = np.frombuffer(joined, dtype=np.uint8).reshape(n, seq_len)

    nodes: Dict[int, FastTreeNode] = {
        i: FastTreeNode(i, name=names[i], height=0.0, size=1) for i in range(n)
    }
    active = set(range(n))
    distances: Dict[Tuple[int, int], float] = {}
    heap: List[Tuple[float, int, int]] = []

    for i in range(n - 1):
        row = np.mean(arr[i + 1:] != arr[i], axis=1)
        for offset, dist in enumerate(row, start=1):
            j = i + offset
            d = float(dist)
            distances[(i, j)] = d
            heapq.heappush(heap, (d, i, j))

    def key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    next_id = n
    while len(active) > 1:
        while heap:
            dist, a, b = heapq.heappop(heap)
            if a in active and b in active and distances.get(key(a, b)) == dist:
                break
        else:
            raise RuntimeError("Fast UPGMA heap exhausted before tree completion")

        na, nb = nodes[a], nodes[b]
        merged_size = na.size + nb.size
        merged_height = max(dist / 2.0, na.height, nb.height)
        if nb.size > na.size:
            na, nb = nb, na
            a, b = b, a
        merged = FastTreeNode(
            next_id,
            left=na,
            right=nb,
            height=merged_height,
            size=merged_size,
        )
        nodes[next_id] = merged

        others = [k for k in active if k not in (a, b)]
        for k in others:
            da = distances.get(key(a, k), 0.0)
            db = distances.get(key(b, k), 0.0)
            nd = (da * na.size + db * nb.size) / merged_size
            distances[key(next_id, k)] = nd
            heapq.heappush(heap, (nd, min(next_id, k), max(next_id, k)))
            distances.pop(key(a, k), None)
            distances.pop(key(b, k), None)
        distances.pop(key(a, b), None)

        active.remove(a)
        active.remove(b)
        active.add(next_id)
        next_id += 1

    return nodes[next(iter(active))]


def _fast_tree_to_newick(node: FastTreeNode, parent_height: Optional[float] = None) -> str:
    branch = 0.0 if parent_height is None else max(parent_height - node.height, 0.0)
    if node.is_leaf():
        return f"{_newick_label(node.name or '')}:{branch:.6f}"
    children = ",".join(_fast_tree_to_newick(c, node.height) for c in node.children())
    if parent_height is None:
        return f"({children});"
    return f"({children}):{branch:.6f}"


def build_fast_upgma_tree_png(
    aligned: Dict[str, str],
    clade_by_name: Dict[str, str],
    out_png: Path,
    out_newick: Path,
    target_names: Optional[Iterable[str]] = None,
    display_note: str = "",
) -> None:
    root = _fast_upgma_root(aligned)
    out_newick.write_text(_fast_tree_to_newick(root), encoding="utf-8")

    clades = sorted(set(clade_by_name.values()))
    cmap = clade_color_map(clades)

    def color_for(name: str) -> str:
        return cmap.get(clade_by_name.get(name, ""), "#5c677d")

    targets = set(target_names or [])

    leaves: List[FastTreeNode] = []

    def assign_y(node: FastTreeNode) -> float:
        if node.is_leaf():
            node.y = float(len(leaves))
            leaves.append(node)
            return node.y
        children = node.children()
        children.sort(key=lambda c: c.size, reverse=True)
        ys = [assign_y(c) for c in children]
        node.y = sum(ys) / len(ys)
        return node.y

    def assign_x(node: FastTreeNode) -> None:
        node.x = max(root.height - node.height, 0.0)
        for child in node.children():
            assign_x(child)

    assign_y(root)
    assign_x(root)

    n = len(leaves)
    xmax = max(root.height, 1e-6)
    fig_h = max(8.0, min(22.0, n * 0.014))
    fig, ax = plt.subplots(figsize=(18, fig_h), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    def draw_edges(node: FastTreeNode) -> None:
        for child in node.children():
            edge_col = color_for(child.name) if child.is_leaf() else "#9ea8b6"
            ax.plot([node.x, node.x], [node.y, child.y], color="#d7dde5", lw=0.55,
                    solid_capstyle="round", alpha=0.78, zorder=1)
            ax.plot([node.x, child.x], [child.y, child.y], color=edge_col, lw=0.9,
                    solid_capstyle="round", alpha=0.88, zorder=2)
            draw_edges(child)

    draw_edges(root)

    for leaf in leaves:
        x, y = leaf.x, leaf.y
        name = leaf.name or ""
        col = color_for(name)
        if name in targets:
            label_left = x > xmax * 0.72
            ax.scatter([x], [y], s=190, marker="*",
                       facecolor="#ffd166", edgecolors="#111827",
                       linewidths=0.9, zorder=6, alpha=1.0)
            ax.annotate(
                f"TARGET: {name}",
                xy=(x, y),
                xytext=(-12, 0) if label_left else (12, 0),
                textcoords="offset points",
                va="center",
                ha="right" if label_left else "left",
                fontsize=8.5,
                fontweight="bold",
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "#fff7d6",
                    "edgecolor": "#111827",
                    "linewidth": 0.8,
                    "alpha": 0.96,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#111827",
                    "linewidth": 0.8,
                    "shrinkA": 0,
                    "shrinkB": 4,
                },
                zorder=7,
                clip_on=False,
            )
        else:
            ax.scatter([x], [y], s=7, facecolor=col, edgecolors="#ffffff",
                       linewidths=0.22, zorder=3, alpha=0.9)

    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(-xmax * 0.03, xmax * 1.36)
    ax.set_yticks([])
    ax.margins(x=0.02, y=0.02)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cfd7de")
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="x", colors="#5c6773", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("average Hamming distance on reference-projected HA", fontsize=10)
    title = "Phylogenetic tree  ·  fast UPGMA full tree"
    if display_note:
        title = f"{title}\n{display_note}"
    ax.set_title(title, fontsize=12, pad=8)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="none",
                          markerfacecolor=cmap[c], markeredgecolor="#f8f9fb",
                          markersize=8, label=c) for c in clades]
    if targets:
        handles.append(
            plt.Line2D(
                [0], [0], marker="*", linestyle="none",
                markerfacecolor="#ffd166", markeredgecolor="#111827",
                markersize=12, label="target sequence",
            )
        )
    legend_cols = 2 if len(handles) > 26 else 1
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              ncol=legend_cols, fontsize=7,
              title="Nextclade clade", title_fontsize=8,
              frameon=True, fancybox=False, framealpha=0.94,
              borderpad=0.45, labelspacing=0.34,
              columnspacing=0.85, handletextpad=0.42)

    plt.subplots_adjust(left=0.03, right=0.82, top=0.93, bottom=0.08)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# ==============================================================================
# ===  표/리포트 저장  ==========================================================
# ==============================================================================

def render_newick_tree_png(
    tree_path: Path,
    tree_format: str,
    clade_by_name: Dict[str, str],
    out_png: Path,
    target_names: Optional[Iterable[str]] = None,
    vaccine_names: Optional[Iterable[str]] = None,
    title: str = "Phylogenetic tree",
    xlabel: str = "branch length",
    display_note: str = "",
    x_by_name: Optional[Dict[str, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    show_clade_bar: bool = True,
    plot_style: str = "dashboard",
    display_max_tips: Optional[int] = None,
    display_branch_cap_years: Optional[float] = None,
) -> Dict[str, object]:
    tree = Phylo.read(str(tree_path), tree_format)
    tree.ladderize()

    clades = sorted(set(clade_by_name.values()))
    cmap = clade_color_map(clades)
    clade_lookup: Dict[str, str] = {}
    for sample_name, clade in clade_by_name.items():
        clade_lookup[tree_label_key(sample_name).lower()] = clade
        clade_lookup[normalize_id(sample_name).lower()] = clade

    def clade_for_name(name: str) -> str:
        return (
            clade_by_name.get(name)
            or clade_lookup.get(tree_label_key(name).lower())
            or clade_lookup.get(normalize_id(name).lower())
            or ""
        )

    def color_for(name: str) -> str:
        return cmap.get(clade_for_name(name), "#5c677d")

    targets = set(target_names or [])
    vaccines = set(vaccine_names or [])
    target_keys = {
        key
        for name in targets
        for key in (tree_label_key(name).lower(), normalize_id(name).lower())
    }
    vaccine_keys = {
        key
        for name in vaccines
        for key in (tree_label_key(name).lower(), normalize_id(name).lower())
    }

    def name_matches(name: str, raw_names: set, keys: set) -> bool:
        return (
            name in raw_names
            or tree_label_key(name).lower() in keys
            or normalize_id(name).lower() in keys
        )

    def is_target_name(name: str) -> bool:
        return name_matches(name, targets, target_keys)

    def is_vaccine_name(name: str) -> bool:
        return name_matches(name, vaccines, vaccine_keys)

    figtree_style = plot_style == "figtree"
    terminals_for_xlim = tree.get_terminals()
    original_tip_count = len(terminals_for_xlim)

    if figtree_style and x_by_name and xlim is None:
        focus_x: List[float] = []
        node_x: List[float] = []
        for terminal in terminals_for_xlim:
            name = terminal.name or ""
            value = x_by_name.get(name)
            if value is not None and math.isfinite(value):
                focus_x.append(float(value))
        for name, value in x_by_name.items():
            if str(name).startswith("NODE_") and math.isfinite(value):
                node_x.append(float(value))
        if len(focus_x) >= 10:
            tip_lo = min(focus_x)
            node_lo = min(node_x) if node_x else tip_lo
            lo = max(min(tip_lo, node_lo), tip_lo - 7.0)
            hi = max(focus_x)
            left = max(math.floor((lo - 1.0) / 5.0) * 5.0, 1800.0)
            right = hi + 1.0
            if right <= left:
                right = left + 1.0
            xlim = (left, right)

    if figtree_style:
        if display_max_tips is None:
            display_max_tips = 260
        if display_max_tips > 0 and original_tip_count > display_max_tips:

            def terminal_date(term) -> Optional[float]:
                value = (x_by_name or {}).get(term.name or "")
                if value is None or not math.isfinite(value):
                    return None
                return float(value)

            def in_display_window(term) -> bool:
                date = terminal_date(term)
                if date is None or not xlim:
                    return True
                return xlim[0] <= date <= xlim[1]

            def evenly_spaced(items: List[object], limit: int) -> List[object]:
                if limit <= 0:
                    return []
                if len(items) <= limit:
                    return list(items)
                if limit == 1:
                    return [items[len(items) // 2]]
                picked: List[object] = []
                seen_names: set = set()
                for idx in range(limit):
                    pos = round(idx * (len(items) - 1) / (limit - 1))
                    item = items[pos]
                    name = item.name or ""
                    if name and name not in seen_names:
                        picked.append(item)
                        seen_names.add(name)
                return picked

            target_clades = {
                clade_for_name(name)
                for name in targets
                if clade_for_name(name) not in (None, "", "unassigned")
            }
            keep_names = {
                term.name or ""
                for term in terminals_for_xlim
                if is_target_name(term.name or "") or is_vaccine_name(term.name or "")
            }
            candidates = [
                term
                for term in terminals_for_xlim
                if (term.name or "") not in keep_names and in_display_window(term)
            ]
            candidates.sort(
                key=lambda term: (
                    terminal_date(term) if terminal_date(term) is not None else -9999.0,
                    tree_label_key(term.name or ""),
                )
            )

            def add_terms(items: List[object]) -> None:
                for term in items:
                    if len(keep_names) >= display_max_tips:
                        break
                    name = term.name or ""
                    if name:
                        keep_names.add(name)

            if target_clades:
                near_target = [
                    term
                    for term in candidates
                    if clade_for_name(term.name or "") in target_clades
                ]
                near_limit = min(max(display_max_tips // 3, 80), display_max_tips // 2)
                add_terms(evenly_spaced(near_target, near_limit))

            year_groups: Dict[int, List[object]] = {}
            for term in candidates:
                name = term.name or ""
                if name in keep_names:
                    continue
                date = terminal_date(term)
                if date is None:
                    continue
                year_groups.setdefault(int(math.floor(date)), []).append(term)
            if year_groups:
                remaining = max(display_max_tips - len(keep_names), 0)
                per_year = max(2, math.ceil(remaining / max(len(year_groups), 1)))
                for year in sorted(year_groups, reverse=True):
                    add_terms(evenly_spaced(year_groups[year], per_year))

            add_terms(evenly_spaced([t for t in candidates if (t.name or "") not in keep_names], display_max_tips))

            if len(keep_names) < min(display_max_tips, original_tip_count):
                all_remaining = [
                    term
                    for term in terminals_for_xlim
                    if (term.name or "") not in keep_names and in_display_window(term)
                ]
                add_terms(evenly_spaced(all_remaining, display_max_tips))

            for terminal in list(tree.get_terminals()):
                name = terminal.name or ""
                if name not in keep_names:
                    tree.prune(terminal)
            terminals_for_xlim = tree.get_terminals()

    depth: Dict[object, float] = {}

    def comp_depth(clade, acc: float) -> None:
        bl = clade.branch_length or 0.0
        if bl < 0:
            bl = 0.0
        depth[clade] = acc + bl
        for child in clade.clades:
            comp_depth(child, depth[clade])

    comp_depth(tree.root, 0.0)

    xcoord: Dict[object, float] = dict(depth)
    if x_by_name:
        explicitly_dated: set = set()

        def date_for_clade(clade) -> Optional[float]:
            name = clade.name or ""
            for key in (name, tree_label_key(name), normalize_id(name)):
                value = x_by_name.get(key)
                if value is not None and math.isfinite(value):
                    return float(value)
            if clade.is_terminal():
                return collection_date_to_decimal_year(extract_collection_date(name))
            return None

        for clade in tree.find_clades():
            date_value = date_for_clade(clade)
            if date_value is not None and math.isfinite(date_value):
                xcoord[clade] = float(date_value)
                explicitly_dated.add(clade)

        def infer_missing_x(clade) -> float:
            if clade in explicitly_dated:
                return xcoord[clade]
            if clade.is_terminal():
                return xcoord.get(clade, depth.get(clade, 0.0))
            child_x = [infer_missing_x(child) for child in clade.clades]
            inferred = min(child_x) if child_x else depth.get(clade, 0.0)
            xcoord[clade] = inferred
            return inferred

        infer_missing_x(tree.root)

        if xlim:
            plot_left, plot_right = xlim
            plot_span = max(plot_right - plot_left, 1.0)
            internal_gap = max(plot_span * 0.004, 0.12)

            def stabilize_calendar_x(clade) -> float:
                if clade.is_terminal():
                    x = xcoord.get(clade, depth.get(clade, plot_left))
                    xcoord[clade] = min(max(x, plot_left), plot_right)
                    return xcoord[clade]
                child_x = [stabilize_calendar_x(child) for child in clade.clades]
                if not child_x:
                    return xcoord.get(clade, plot_left)
                child_min = min(child_x)
                raw_x = xcoord.get(clade, child_min - internal_gap)
                if raw_x < plot_left or raw_x > child_min:
                    raw_x = child_min - internal_gap
                xcoord[clade] = min(max(raw_x, plot_left), plot_right)
                return xcoord[clade]

            stabilize_calendar_x(tree.root)

    if figtree_style and x_by_name:
        if display_branch_cap_years is None:
            display_branch_cap_years = 0.65
        if display_branch_cap_years > 0:
            axis_left = xlim[0] if xlim else None
            min_gap = max(display_branch_cap_years * 0.035, 0.05)

            def compress_horizontal_branches(clade) -> float:
                if clade.is_terminal():
                    return xcoord.get(clade, depth.get(clade, 0.0))
                child_x = [compress_horizontal_branches(child) for child in clade.clades]
                if not child_x:
                    return xcoord.get(clade, depth.get(clade, 0.0))
                child_min = min(child_x)
                x = xcoord.get(clade, depth.get(clade, child_min))
                if child_min - x > display_branch_cap_years:
                    x = child_min - display_branch_cap_years
                if x > child_min - min_gap:
                    x = child_min - min_gap
                if axis_left is not None:
                    x = max(x, axis_left)
                xcoord[clade] = x
                return x

            compress_horizontal_branches(tree.root)

    trunk_edges: set = set()
    trunk_terminal_name = ""

    if figtree_style and x_by_name:
        max_tip_date_cache: Dict[object, float] = {}
        median_tip_date_cache: Dict[object, float] = {}
        tip_count_cache: Dict[object, int] = {}

        def terminal_calendar_x(clade) -> float:
            name = clade.name or ""
            for key in (name, tree_label_key(name), normalize_id(name)):
                value = x_by_name.get(key) if x_by_name else None
                if value is not None and math.isfinite(value):
                    return float(value)
            date_value = collection_date_to_decimal_year(extract_collection_date(name))
            if date_value is not None and math.isfinite(date_value):
                return float(date_value)
            return float("-inf")

        def max_tip_date(clade) -> float:
            cached = max_tip_date_cache.get(clade)
            if cached is not None:
                return cached
            if clade.is_terminal():
                result = terminal_calendar_x(clade)
            else:
                child_values = [max_tip_date(child) for child in clade.clades]
                result = max(child_values) if child_values else terminal_calendar_x(clade)
            max_tip_date_cache[clade] = result
            return result

        def median_tip_date(clade) -> float:
            cached = median_tip_date_cache.get(clade)
            if cached is not None:
                return cached
            if clade.is_terminal():
                result = terminal_calendar_x(clade)
            else:
                values = [
                    median_tip_date(child)
                    for child in clade.clades
                    if math.isfinite(median_tip_date(child))
                ]
                result = float(np.median(np.array(values, dtype=float))) if values else terminal_calendar_x(clade)
            median_tip_date_cache[clade] = result
            return result

        def tip_count(clade) -> int:
            cached = tip_count_cache.get(clade)
            if cached is not None:
                return cached
            if clade.is_terminal():
                result = 1
            else:
                result = sum(tip_count(child) for child in clade.clades)
            tip_count_cache[clade] = result
            return result

        # Approximate the influenza trunk as the child path that repeatedly
        # reaches the most recent observed tip.  This is a visualization aid,
        # not a biological clade reassignment.
        current = tree.root
        while current.clades:
            next_child = max(
                current.clades,
                key=lambda child: (
                    max_tip_date(child),
                    median_tip_date(child),
                    tip_count(child),
                ),
            )
            trunk_edges.add((current, next_child))
            current = next_child
        trunk_terminal_name = current.name or ""

        def sort_by_trunk_calendar(clade) -> float:
            if clade.is_terminal():
                return terminal_calendar_x(clade)
            for child in clade.clades:
                sort_by_trunk_calendar(child)
            trunk_child = next(
                (child for child in clade.clades if (clade, child) in trunk_edges),
                None,
            )
            side_children = [child for child in clade.clades if child is not trunk_child]
            side_children = sorted(
                side_children,
                key=lambda child: (
                    max_tip_date(child),
                    median_tip_date(child),
                    tip_count(child),
                ),
                reverse=True,
            )
            clade.clades = ([trunk_child] if trunk_child is not None else []) + side_children
            return median_tip_date(clade)

        sort_by_trunk_calendar(tree.root)
        terminals_for_xlim = tree.get_terminals()

    terminals = terminals_for_xlim
    n = len(terminals)
    ypos: Dict[object, float] = {t: i for i, t in enumerate(terminals)}

    def assign_y(clade) -> float:
        if clade.is_terminal():
            return ypos[clade]
        child_y = {child: assign_y(child) for child in clade.clades}
        ys = list(child_y.values())
        trunk_child = next(
            (child for child in clade.clades if (clade, child) in trunk_edges),
            None,
        )
        if figtree_style and trunk_child is not None:
            mean_y = sum(ys) / len(ys)
            ypos[clade] = child_y[trunk_child] * 0.72 + mean_y * 0.28
        else:
            ypos[clade] = sum(ys) / len(ys)
        return ypos[clade]

    assign_y(tree.root)

    if figtree_style:
        x_values = list(xcoord.values())
        xmax = max(x_values) if x_values else 1.0
        xmin = min(x_values) if x_values else 0.0
        if xlim:
            axis_left, axis_right = xlim
        elif x_by_name:
            span = max(xmax - xmin, 1.0)
            axis_left, axis_right = xmin - span * 0.02, xmax + span * 0.06
        else:
            axis_left, axis_right = -xmax * 0.03, xmax * 1.18
        axis_span = max(axis_right - axis_left, 1.0)

        def short_tree_label(name: str) -> str:
            text = re.sub(r"_?EPI_ISL_\d+.*$", "", name)
            text = re.sub(r"_?(?:19|20)\d{2}[-_]\d{2}[-_]\d{2}$", "", text)
            text = text.replace("_", "/")
            return text[:42]

        fig = plt.figure(figsize=(11.2, 13.4), dpi=220)
        fig.patch.set_facecolor("#ffffff")
        ax = fig.add_axes([0.055, 0.235, 0.78, 0.715])
        focus_left = fig.add_axes([0.055, 0.055, 0.37, 0.125])
        focus_right = fig.add_axes([0.465, 0.055, 0.37, 0.125])
        legend_ax = fig.add_axes([0.855, 0.055, 0.12, 0.895])
        legend_ax.axis("off")

        ax.set_facecolor("#ffffff")
        for focus_axis in (focus_left, focus_right):
            focus_axis.set_facecolor("#ffffff")
            focus_axis.set_xticks([])
            focus_axis.set_yticks([])
            for spine in focus_axis.spines.values():
                spine.set_color("#d0d4d9")
                spine.set_linewidth(0.65)

        def draw_main_edges(clade) -> None:
            x0 = xcoord[clade]
            y0 = ypos[clade]
            for child in clade.clades:
                x1 = xcoord[child]
                y1 = ypos[child]
                length = max(x1 - x0, 0.0)
                is_trunk_edge = (clade, child) in trunk_edges
                if is_trunk_edge:
                    vertical_col, vertical_alpha, vertical_lw = "#2f343a", 0.88, 0.36
                    col, alpha, lw, zorder = "#111111", 0.96, 0.72, 5
                elif length > 4.0:
                    vertical_col, vertical_alpha, vertical_lw = "#9ca3ad", 0.38, 0.24
                    col, alpha, lw, zorder = "#a9b1bb", 0.58, 0.30, 2
                elif length > 1.4:
                    vertical_col, vertical_alpha, vertical_lw = "#8f98a3", 0.42, 0.25
                    col, alpha, lw, zorder = "#7d8792", 0.68, 0.34, 3
                else:
                    vertical_col, vertical_alpha, vertical_lw = "#77818d", 0.48, 0.27
                    col, alpha, lw, zorder = "#525d69", 0.76, 0.36, 3
                ax.plot([x0, x0], [y0, y1], color=vertical_col, lw=vertical_lw,
                        solid_capstyle="butt", alpha=vertical_alpha, zorder=1)
                if is_trunk_edge and length > 2.5:
                    tail = max(display_branch_cap_years or 0.65, 0.65)
                    join_x = max(x0, x1 - tail)
                    ax.plot([x0, join_x], [y1, y1], color="#8f98a3", lw=0.26,
                            solid_capstyle="butt", alpha=0.42, zorder=2)
                    ax.plot([join_x, x1], [y1, y1], color=col, lw=lw,
                            solid_capstyle="butt", alpha=alpha, zorder=zorder)
                else:
                    ax.plot([x0, x1], [y1, y1], color=col, lw=lw,
                            solid_capstyle="butt", alpha=alpha, zorder=zorder)
                draw_main_edges(child)

        draw_main_edges(tree.root)

        focus_terms = [
            terminal for terminal in terminals
            if is_target_name(terminal.name or "") or is_vaccine_name(terminal.name or "")
        ]
        target_terms = [term for term in focus_terms if is_target_name(term.name or "")]
        vaccine_terms = [term for term in focus_terms if is_vaccine_name(term.name or "")]

        for terminal in terminals:
            name = terminal.name or ""
            x, y = xcoord[terminal], ypos[terminal]
            if is_target_name(name):
                ax.scatter([x], [y], s=24, marker="o",
                           facecolor="#0057ff", edgecolors="#ffffff",
                           linewidths=0.55, zorder=8)
                ax.annotate(short_tree_label(name), xy=(x, y), xytext=(5, 0),
                            textcoords="offset points", va="center", ha="left",
                            fontsize=5.2, color="#0057ff", clip_on=False, zorder=9)
            elif is_vaccine_name(name):
                ax.scatter([x], [y], s=30, marker="^",
                           facecolor="#e53935", edgecolors="#ffffff",
                           linewidths=0.5, zorder=8)

        ax.set_xlim(axis_left, axis_right)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks([])
        tick_start = int(math.ceil(axis_left))
        tick_end = int(math.floor(axis_right))
        if tick_end >= tick_start:
            tick_span = tick_end - tick_start
            tick_step = 1 if tick_span <= 18 else (2 if tick_span <= 32 else 5)
            ticks = list(range(tick_start, tick_end + 1, tick_step))
            ax.set_xticks(ticks)
            for tick in ticks:
                ax.axvline(tick, color="#f0f2f4", lw=0.45, zorder=0)
        ax.tick_params(axis="x", colors="#4b5563", labelsize=6, length=2.2, width=0.5)
        ax.tick_params(axis="y", length=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#111111")
        ax.spines["bottom"].set_linewidth(0.45)

        parent_by_clade: Dict[object, object] = {}
        for parent in tree.find_clades():
            for child in parent.clades:
                parent_by_clade[child] = parent

        terminals_under_cache: Dict[object, List[object]] = {}

        def terminals_under(clade) -> List[object]:
            cached = terminals_under_cache.get(clade)
            if cached is not None:
                return cached
            if clade.is_terminal():
                result = [clade]
            else:
                result = []
                for child in clade.clades:
                    result.extend(terminals_under(child))
            terminals_under_cache[clade] = result
            return result

        def ancestor_chain(clade) -> List[object]:
            chain = [clade]
            while chain[-1] in parent_by_clade:
                chain.append(parent_by_clade[chain[-1]])
            return chain

        def choose_neighborhood_roots(seed_terms: List[object]) -> List[object]:
            roots: List[object] = []
            for term in seed_terms:
                selected = term
                for ancestor in ancestor_chain(term):
                    count = len(terminals_under(ancestor))
                    if count > 58:
                        break
                    selected = ancestor
                    if count >= 18:
                        break
                roots.append(selected)
            unique_roots: List[object] = []
            for root in roots:
                if root in unique_roots:
                    continue
                root_ancestors = set(ancestor_chain(root)[1:])
                if any(other in root_ancestors for other in roots if other is not root):
                    continue
                unique_roots.append(root)
            return unique_roots or seed_terms[:1]

        def local_layout(roots: List[object]) -> Tuple[List[object], Dict[object, float], Dict[object, float]]:
            local_terms: List[object] = []
            seen: set = set()
            for root in roots:
                for term in terminals_under(root):
                    if term not in seen:
                        local_terms.append(term)
                        seen.add(term)
            local_terms.sort(key=lambda term: ypos[term])
            local_y: Dict[object, float] = {term: float(idx) for idx, term in enumerate(local_terms)}

            def assign_local_y(clade) -> Optional[float]:
                if clade.is_terminal():
                    return local_y.get(clade)
                child_ys = [
                    value for child in clade.clades
                    if (value := assign_local_y(child)) is not None
                ]
                if not child_ys:
                    return None
                local_y[clade] = sum(child_ys) / len(child_ys)
                return local_y[clade]

            for root in roots:
                assign_local_y(root)

            local_x = dict(xcoord)
            cap = max(display_branch_cap_years or 0.65, 0.45)
            min_gap = max(cap * 0.04, 0.04)

            def compress_local_x(clade) -> float:
                if clade.is_terminal():
                    return local_x.get(clade, xcoord.get(clade, 0.0))
                child_x = [compress_local_x(child) for child in clade.clades]
                if not child_x:
                    return local_x.get(clade, xcoord.get(clade, 0.0))
                child_min = min(child_x)
                x = local_x.get(clade, xcoord.get(clade, child_min))
                if child_min - x > cap:
                    x = child_min - cap
                if x > child_min - min_gap:
                    x = child_min - min_gap
                local_x[clade] = x
                return x

            for root in roots:
                compress_local_x(root)
            return local_terms, local_y, local_x

        def draw_local_edges(axis, clade, local_y: Dict[object, float],
                             local_x: Dict[object, float]) -> None:
            if clade not in local_y:
                return
            x0 = local_x[clade]
            y0 = local_y[clade]
            for child in clade.clades:
                if child not in local_y:
                    continue
                x1 = local_x[child]
                y1 = local_y[child]
                axis.plot([x0, x0], [y0, y1], color="#30343b", lw=0.55,
                          solid_capstyle="butt", alpha=0.88, zorder=1)
                axis.plot([x0, x1], [y1, y1], color="#30343b", lw=0.55,
                          solid_capstyle="butt", alpha=0.88, zorder=2)
                draw_local_edges(axis, child, local_y, local_x)

        def draw_focus_panel(axis, seed_terms: List[object], fallback_terms: List[object],
                             title_text: str, marker_kind: str) -> None:
            seeds = seed_terms or fallback_terms[:1]
            if not seeds:
                axis.axis("off")
                return
            roots = choose_neighborhood_roots(seeds)
            local_terms, local_y, local_x = local_layout(roots)
            for root in roots:
                draw_local_edges(axis, root, local_y, local_x)
            for term in local_terms:
                name = term.name or ""
                x, y = local_x[term], local_y[term]
                if is_target_name(name):
                    axis.scatter([x], [y], s=34, marker="o",
                                 facecolor="#0057ff", edgecolors="#ffffff",
                                 linewidths=0.6, zorder=8)
                    axis.annotate(short_tree_label(name), xy=(x, y), xytext=(5, 0),
                                  textcoords="offset points", va="center", ha="left",
                                  fontsize=5.4, color="#0057ff", clip_on=False, zorder=9)
                elif is_vaccine_name(name):
                    axis.scatter([x], [y], s=42, marker="^",
                                 facecolor="#e53935", edgecolors="#ffffff",
                                 linewidths=0.55, zorder=8)
                    axis.annotate(short_tree_label(name), xy=(x, y), xytext=(5, 0),
                                  textcoords="offset points", va="center", ha="left",
                                  fontsize=5.4, color="#1f2937", clip_on=False, zorder=9)
            if marker_kind == "target":
                title_color = "#0057ff"
            else:
                title_color = "#e53935"
            axis.text(0.01, 0.96, title_text, transform=axis.transAxes,
                      va="top", ha="left", fontsize=7.4, color=title_color,
                      fontweight="bold")
            xvals = [local_x[item] for item in local_y if item in local_x]
            xmin_local, xmax_local = min(xvals), max(xvals)
            xpad = max((xmax_local - xmin_local) * 0.08, 0.16)
            axis.set_xlim(xmin_local - xpad, xmax_local + xpad)
            axis.set_ylim(len(local_terms) - 0.5, -0.5)
            axis.margins(x=0.02, y=0.05)

        draw_focus_panel(focus_left, target_terms, focus_terms, "target neighborhood", "target")
        draw_focus_panel(focus_right, vaccine_terms, focus_terms, "vaccine strains", "vaccine")

        def draw_clade_brackets() -> None:
            ordered_terms = sorted(terminals, key=lambda item: ypos[item])
            if not ordered_terms:
                return
            runs: List[Tuple[str, float, float, int]] = []
            current_clade = clade_for_name(ordered_terms[0].name or "") or "unassigned"
            run_start = ypos[ordered_terms[0]]
            run_end = run_start
            run_count = 0
            for terminal in ordered_terms:
                clade = clade_for_name(terminal.name or "") or "unassigned"
                y = ypos[terminal]
                if clade != current_clade and run_count:
                    runs.append((current_clade, run_start, run_end, run_count))
                    current_clade = clade
                    run_start = y
                    run_count = 0
                run_end = y
                run_count += 1
            runs.append((current_clade, run_start, run_end, run_count))
            bracket_x = axis_right + axis_span * 0.018
            tick = axis_span * 0.007
            label_runs = [
                run for run in runs
                if run[0] != "unassigned" and run[3] >= max(10, n // 95)
            ]
            label_runs = sorted(label_runs, key=lambda run: (-run[3], run[1]))[:22]
            used_y: List[float] = []
            for clade, y0, y1, _count in sorted(label_runs, key=lambda run: run[1]):
                label_y = (y0 + y1) / 2
                if any(abs(label_y - prev) < 12 for prev in used_y):
                    continue
                used_y.append(label_y)
                ax.plot([bracket_x, bracket_x], [y0 - 0.4, y1 + 0.4],
                        color="#111111", lw=0.45, clip_on=False, zorder=9)
                ax.plot([bracket_x - tick, bracket_x], [y0 - 0.4, y0 - 0.4],
                        color="#111111", lw=0.45, clip_on=False, zorder=9)
                ax.plot([bracket_x - tick, bracket_x], [y1 + 0.4, y1 + 0.4],
                        color="#111111", lw=0.45, clip_on=False, zorder=9)
                ax.text(bracket_x + tick * 0.55, label_y, clade,
                        va="center", ha="left", fontsize=5.0,
                        color="#111111", clip_on=False, zorder=10)

        draw_clade_brackets()

        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.scatter([0.08], [0.94], s=34, marker="o",
                          facecolor="#0057ff", edgecolors="#ffffff", linewidths=0.6,
                          transform=legend_ax.transAxes)
        legend_ax.text(0.17, 0.94, "target", va="center", ha="left",
                       fontsize=7.0, color="#111827", transform=legend_ax.transAxes)
        legend_ax.scatter([0.08], [0.90], s=44, marker="^",
                          facecolor="#e53935", edgecolors="#ffffff", linewidths=0.6,
                          transform=legend_ax.transAxes)
        legend_ax.text(0.17, 0.90, "vaccine", va="center", ha="left",
                       fontsize=7.0, color="#111827", transform=legend_ax.transAxes)
        legend_ax.plot([0.06, 0.19], [0.86, 0.86], color="#111111", lw=1.0,
                       solid_capstyle="butt", transform=legend_ax.transAxes)
        legend_ax.text(0.24, 0.86, "backbone", va="center", ha="left",
                       fontsize=6.8, color="#111827", transform=legend_ax.transAxes)
        legend_ax.text(0.08, 0.82, f"{n:,} displayed tips", va="center",
                       ha="left", fontsize=6.6, color="#4b5563",
                       transform=legend_ax.transAxes)
        if trunk_terminal_name:
            legend_ax.text(0.08, 0.78, short_tree_label(trunk_terminal_name),
                           va="center", ha="left", fontsize=5.4,
                           color="#6b7280", transform=legend_ax.transAxes)

        fig.savefig(out_png, dpi=220, bbox_inches=None)
        plt.close(fig)
        return {
            "tree_display_original_tips": original_tip_count,
            "tree_display_tips": n,
            "tree_display_max_tips": display_max_tips if figtree_style else 0,
            "tree_display_branch_cap_years": (
                display_branch_cap_years if figtree_style and x_by_name else 0
            ),
            "tree_display_sampled": bool(figtree_style and display_max_tips and n < original_tip_count),
        }

    x_values = list(xcoord.values())
    xmax = max(x_values) if x_values else 1.0
    xmin = min(x_values) if x_values else 0.0
    xmax = max(xmax, xmin + 1e-6)
    if figtree_style:
        fig_h = max(10.5, min(17.0, n * 0.026))
        fig_w = 8.0
    else:
        fig_h = max(8.0, min(22.0, n * 0.014))
        fig_w = 18.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220 if figtree_style else 180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    def draw_edges(clade, axis, lw_scale: float = 1.0, fade_long: bool = True) -> None:
        x0 = xcoord[clade]
        y0 = ypos[clade]
        for child in clade.clades:
            x1 = xcoord[child]
            y1 = ypos[child]
            if figtree_style:
                axis.plot([x0, x0], [y0, y1], color="#1f1f1f", lw=0.42 * lw_scale,
                          solid_capstyle="butt", alpha=0.9, zorder=1)
                solid_tail = display_branch_cap_years or 0
                if fade_long and solid_tail > 0 and x1 - x0 > solid_tail * 1.25:
                    join_x = max(x0, x1 - solid_tail)
                    axis.plot([x0, join_x], [y1, y1], color="#b9b9b9", lw=0.24 * lw_scale,
                              solid_capstyle="butt", alpha=0.34, zorder=1)
                    axis.plot([join_x, x1], [y1, y1], color="#1f1f1f", lw=0.4 * lw_scale,
                              solid_capstyle="butt", alpha=0.9, zorder=2)
                else:
                    axis.plot([x0, x1], [y1, y1], color="#1f1f1f", lw=0.42 * lw_scale,
                              solid_capstyle="butt", alpha=0.9, zorder=2)
            else:
                edge_col = color_for(child.name) if child.is_terminal() else "#9ea8b6"
                axis.plot([x0, x0], [y0, y1], color="#d7dde5", lw=0.55 * lw_scale,
                          solid_capstyle="round", alpha=0.78, zorder=1)
                axis.plot([x0, x1], [y1, y1], color=edge_col, lw=0.9 * lw_scale,
                          solid_capstyle="round", alpha=0.88, zorder=2)
            draw_edges(child, axis, lw_scale=lw_scale, fade_long=fade_long)

    draw_edges(tree.root, ax)

    for terminal in terminals:
        x, y = xcoord[terminal], ypos[terminal]
        name = terminal.name or ""
        col = color_for(name)
        if figtree_style:
            if is_vaccine_name(name):
                ax.scatter([x], [y], s=28, marker="^",
                           facecolor="#ff1f1f", edgecolors="#ff1f1f",
                           linewidths=0.4, zorder=7, alpha=0.96)
            elif is_target_name(name):
                ax.scatter([x], [y], s=20, marker="o",
                           facecolor="#003cff", edgecolors="#003cff",
                           linewidths=0.3, zorder=7, alpha=0.96)
                ax.annotate(
                    name,
                    xy=(x, y),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=5.6,
                    color="#003cff",
                    clip_on=False,
                    zorder=8,
                )
        elif name in targets:
            label_left = False
            if xlim:
                axis_left, axis_right = xlim
                label_left = x > axis_left + (axis_right - axis_left) * 0.72
            ax.scatter([x], [y], s=190, marker="*",
                       facecolor="#ffd166", edgecolors="#111827",
                       linewidths=0.9, zorder=6, alpha=1.0)
            ax.annotate(
                f"TARGET: {name}",
                xy=(x, y),
                xytext=(-12, 0) if label_left else (12, 0),
                textcoords="offset points",
                va="center",
                ha="right" if label_left else "left",
                fontsize=8.5,
                fontweight="bold",
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "#fff7d6",
                    "edgecolor": "#111827",
                    "linewidth": 0.8,
                    "alpha": 0.96,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#111827",
                    "linewidth": 0.8,
                    "shrinkA": 0,
                    "shrinkB": 4,
                },
                zorder=7,
                clip_on=False,
            )
        else:
            ax.scatter([x], [y], s=7, facecolor=col, edgecolors="#ffffff",
                       linewidths=0.22, zorder=3, alpha=0.9)

    ax.set_ylim(n - 0.5, -0.5)
    if xlim:
        x_min, x_max = xlim
        ax.set_xlim(x_min, x_max)
    elif x_by_name:
        span = max(xmax - xmin, 1.0)
        ax.set_xlim(xmin - span * 0.03, xmax + span * 0.12)
    else:
        ax.set_xlim(-xmax * 0.03, xmax * 1.36)

    focus_insets_drawn = False
    if figtree_style:
        def short_tree_label(name: str) -> str:
            text = re.sub(r"_?EPI_ISL_\d+.*$", "", name)
            text = re.sub(r"_?(?:19|20)\d{2}[-_]\d{2}[-_]\d{2}$", "", text)
            text = text.replace("_", "/")
            return text[:42]

        def draw_focus_markers(
            axis,
            label_markers: bool,
            marker_scale: float = 1.0,
            y_map: Optional[Dict[object, float]] = None,
            x_map: Optional[Dict[object, float]] = None,
            allowed_terms: Optional[set] = None,
        ) -> None:
            for terminal in terminals:
                if allowed_terms is not None and terminal not in allowed_terms:
                    continue
                if y_map is not None and terminal not in y_map:
                    continue
                x = (x_map or xcoord)[terminal]
                y = (y_map or ypos)[terminal]
                name = terminal.name or ""
                if is_vaccine_name(name):
                    axis.scatter([x], [y], s=32 * marker_scale, marker="^",
                                 facecolor="#ff1f1f", edgecolors="#ff1f1f",
                                 linewidths=0.35, zorder=8, alpha=0.98)
                    if label_markers:
                        axis.annotate(short_tree_label(name), xy=(x, y), xytext=(4, 0),
                                      textcoords="offset points", va="center", ha="left",
                                      fontsize=4.8 * marker_scale, color="#111111",
                                      clip_on=False, zorder=9)
                elif is_target_name(name):
                    axis.scatter([x], [y], s=22 * marker_scale, marker="o",
                                 facecolor="#003cff", edgecolors="#003cff",
                                 linewidths=0.25, zorder=8, alpha=0.98)
                    if label_markers:
                        axis.annotate(short_tree_label(name), xy=(x, y), xytext=(4, 0),
                                      textcoords="offset points", va="center", ha="left",
                                      fontsize=4.8 * marker_scale, color="#003cff",
                                      clip_on=False, zorder=9)

        parent_by_clade: Dict[object, object] = {}
        for parent in tree.find_clades():
            for child in parent.clades:
                parent_by_clade[child] = parent

        terminals_under_cache: Dict[object, List[object]] = {}

        def terminals_under(clade) -> List[object]:
            cached = terminals_under_cache.get(clade)
            if cached is not None:
                return cached
            if clade.is_terminal():
                result = [clade]
            else:
                result = []
                for child in clade.clades:
                    result.extend(terminals_under(child))
            terminals_under_cache[clade] = result
            return result

        def ancestor_chain(clade) -> List[object]:
            chain = [clade]
            while chain[-1] in parent_by_clade:
                chain.append(parent_by_clade[chain[-1]])
            return chain

        def choose_focus_roots(group: List[object]) -> List[object]:
            min_terms = 14
            max_terms = 70
            if len(group) > 1:
                common = set(ancestor_chain(group[0]))
                for term in group[1:]:
                    common &= set(ancestor_chain(term))
                if common:
                    mrca = max(common, key=lambda item: len(ancestor_chain(item)))
                    if len(terminals_under(mrca)) <= max_terms:
                        return [mrca]

            roots: List[object] = []
            for term in group:
                selected = term
                for ancestor in ancestor_chain(term):
                    count = len(terminals_under(ancestor))
                    if count > max_terms:
                        break
                    selected = ancestor
                    if count >= min_terms:
                        break
                roots.append(selected)

            unique_roots: List[object] = []
            for root in roots:
                if root in unique_roots:
                    continue
                root_ancestors = set(ancestor_chain(root)[1:])
                if any(other in root_ancestors for other in roots if other is not root):
                    continue
                unique_roots.append(root)
            return unique_roots or group

        def assign_local_y_for_roots(roots: List[object]) -> Tuple[List[object], Dict[object, float]]:
            local_terms: List[object] = []
            seen: set = set()
            for root in roots:
                for term in terminals_under(root):
                    if term not in seen:
                        local_terms.append(term)
                        seen.add(term)
            local_terms.sort(key=lambda term: ypos[term])
            local_y: Dict[object, float] = {term: float(idx) for idx, term in enumerate(local_terms)}

            def assign_local_y(clade) -> Optional[float]:
                if clade.is_terminal():
                    return local_y.get(clade)
                child_ys = [
                    value for child in clade.clades
                    if (value := assign_local_y(child)) is not None
                ]
                if not child_ys:
                    return None
                local_y[clade] = sum(child_ys) / len(child_ys)
                return local_y[clade]

            for root in roots:
                assign_local_y(root)
            return local_terms, local_y

        def local_x_for_roots(roots: List[object]) -> Dict[object, float]:
            local_x = dict(xcoord)
            cap = max(display_branch_cap_years or 0.65, 0.35)
            min_gap = max(cap * 0.035, 0.035)

            def compress(clade) -> float:
                if clade.is_terminal():
                    return local_x.get(clade, xcoord.get(clade, 0.0))
                child_x = [compress(child) for child in clade.clades]
                if not child_x:
                    return local_x.get(clade, xcoord.get(clade, 0.0))
                child_min = min(child_x)
                x = local_x.get(clade, xcoord.get(clade, child_min))
                if child_min - x > cap:
                    x = child_min - cap
                if x > child_min - min_gap:
                    x = child_min - min_gap
                local_x[clade] = x
                return x

            for root in roots:
                compress(root)
            return local_x

        def draw_local_edges(clade, axis, local_y: Dict[object, float],
                             local_x: Dict[object, float]) -> None:
            if clade not in local_y:
                return
            x0 = local_x[clade]
            y0 = local_y[clade]
            for child in clade.clades:
                if child not in local_y:
                    continue
                x1 = local_x[child]
                y1 = local_y[child]
                axis.plot([x0, x0], [y0, y1], color="#1f1f1f", lw=0.52,
                          solid_capstyle="butt", alpha=0.92, zorder=1)
                axis.plot([x0, x1], [y1, y1], color="#1f1f1f", lw=0.52,
                          solid_capstyle="butt", alpha=0.92, zorder=2)
                draw_local_edges(child, axis, local_y, local_x)

        focus_terms = [
            terminal for terminal in terminals
            if is_target_name(terminal.name or "") or is_vaccine_name(terminal.name or "")
        ]
        if focus_terms:
            axis_left, axis_right = ax.get_xlim()
            sorted_focus = sorted(focus_terms, key=lambda item: ypos[item])
            groups: List[List[object]] = []
            current_group: List[object] = []
            split_gap = max(70.0, min(140.0, n * 0.10))
            prev_y: Optional[float] = None
            for term in sorted_focus:
                y = ypos[term]
                if current_group and prev_y is not None and y - prev_y > split_gap:
                    groups.append(current_group)
                    current_group = []
                current_group.append(term)
                prev_y = y
            if current_group:
                groups.append(current_group)

            groups = sorted(
                groups,
                key=lambda group: (
                    not any(is_target_name(term.name or "") for term in group),
                    -len(group),
                    min(ypos[term] for term in group),
                ),
            )[:2]
            groups = sorted(groups, key=lambda group: min(ypos[term] for term in group))
            inset_positions = (
                [(0.045, 0.62, 0.42, 0.30), (0.045, 0.31, 0.42, 0.24)]
                if len(groups) > 1 else
                [(0.045, 0.40, 0.42, 0.36)]
            )
            focus_insets_drawn = True
            for group, position in zip(groups, inset_positions):
                focus_roots = choose_focus_roots(group)
                local_terms, local_y = assign_local_y_for_roots(focus_roots)
                local_x = local_x_for_roots(focus_roots)
                if not local_terms:
                    continue

                global_ys = [ypos[term] for term in local_terms]
                y0 = max(min(global_ys) - 0.8, -0.5)
                y1 = min(max(global_ys) + 0.8, n - 0.5)
                global_xs = [xcoord[term] for term in local_terms]
                main_x0 = max(axis_left, min(global_xs) - 0.25)
                main_x1 = min(axis_right, max(global_xs) + 0.25)
                local_clades = list(local_y.keys())
                local_xs = [local_x[clade] for clade in local_clades if clade in local_x]
                ix0 = min(local_xs) - 0.10
                ix1 = max(local_xs) + 0.20
                if ix1 - ix0 < 1.5:
                    mid = (ix0 + ix1) / 2
                    ix0 = mid - 0.75
                    ix1 = mid + 0.75

                rect = matplotlib.patches.Rectangle(
                    (main_x0, y0), main_x1 - main_x0, y1 - y0,
                    fill=False, edgecolor="#111111", linewidth=0.55,
                    linestyle=(0, (3, 2)), alpha=0.85,
                    clip_on=False, zorder=9,
                )
                ax.add_patch(rect)

                inset_ax = fig.add_axes(position)
                inset_ax.set_facecolor("#ffffff")
                for root in focus_roots:
                    draw_local_edges(root, inset_ax, local_y, local_x)
                draw_focus_markers(
                    inset_ax,
                    label_markers=True,
                    marker_scale=1.05,
                    y_map=local_y,
                    x_map=local_x,
                    allowed_terms=set(local_terms),
                )
                inset_ax.set_xlim(ix0, ix1)
                inset_ax.set_ylim(len(local_terms) - 0.5, -0.5)
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])
                for spine in inset_ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#111111")
                    spine.set_linewidth(0.55)
                    spine.set_linestyle((0, (3, 2)))

                for inset_y, main_y in ((-0.5, y0), (len(local_terms) - 0.5, y1)):
                    connector = matplotlib.patches.ConnectionPatch(
                        xyA=(ix1, inset_y), coordsA=inset_ax.transData,
                        xyB=(main_x0, main_y), coordsB=ax.transData,
                        color="#111111", linewidth=0.45,
                        linestyle=(0, (3, 3)), alpha=0.75,
                        zorder=4,
                    )
                    fig.add_artist(connector)

    if show_clade_bar and not figtree_style and terminals:
        axis_left, axis_right = ax.get_xlim()
        axis_span = max(axis_right - axis_left, 1.0)
        bar_x = axis_right + axis_span * 0.012
        text_x = axis_right + axis_span * 0.019
        ordered_terms = sorted(terminals, key=lambda item: ypos[item])
        runs: List[Tuple[str, float, float, int]] = []
        current_clade = clade_by_name.get(ordered_terms[0].name or "", "unassigned") or "unassigned"
        run_start = ypos[ordered_terms[0]]
        run_end = run_start
        run_count = 0
        for terminal in ordered_terms:
            clade = clade_by_name.get(terminal.name or "", "unassigned") or "unassigned"
            y = ypos[terminal]
            if clade != current_clade and run_count:
                runs.append((current_clade, run_start, run_end, run_count))
                current_clade = clade
                run_start = y
                run_count = 0
            run_end = y
            run_count += 1
        runs.append((current_clade, run_start, run_end, run_count))

        label_run_for_clade: Dict[str, int] = {}
        for idx, (clade, _y0, _y1, count) in enumerate(runs):
            if count < 3:
                continue
            prev = label_run_for_clade.get(clade)
            if prev is None or count > runs[prev][3]:
                label_run_for_clade[clade] = idx
        label_run_indices = set(label_run_for_clade.values())

        labeled_runs = 0
        for idx, (clade, y0, y1, count) in enumerate(runs):
            col = cmap.get(clade, "#9aa5b1")
            ax.plot(
                [bar_x, bar_x],
                [y0 - 0.45, y1 + 0.45],
                color=col,
                lw=4.3,
                solid_capstyle="butt",
                alpha=0.96,
                zorder=5,
                clip_on=False,
            )
            if idx in label_run_indices and labeled_runs < 48:
                ax.text(
                    text_x,
                    (y0 + y1) / 2,
                    clade,
                    va="center",
                    ha="left",
                    fontsize=6.3,
                    color="#111827",
                    clip_on=False,
                )
                labeled_runs += 1
    if figtree_style and terminals:
        axis_left, axis_right = ax.get_xlim()
        axis_span = max(axis_right - axis_left, 1.0)
        bracket_x = axis_right + axis_span * 0.022
        bracket_tick = axis_span * 0.010
        ordered_terms = sorted(terminals, key=lambda item: ypos[item])
        runs: List[Tuple[str, float, float, int]] = []
        current_clade = clade_for_name(ordered_terms[0].name or "") or "unassigned"
        run_start = ypos[ordered_terms[0]]
        run_end = run_start
        run_count = 0
        for terminal in ordered_terms:
            clade = clade_for_name(terminal.name or "") or "unassigned"
            y = ypos[terminal]
            if clade != current_clade and run_count:
                runs.append((current_clade, run_start, run_end, run_count))
                current_clade = clade
                run_start = y
                run_count = 0
            run_end = y
            run_count += 1
        runs.append((current_clade, run_start, run_end, run_count))

        labeled_y: List[float] = []
        large_runs = [
            run for run in runs
            if run[0] != "unassigned" and run[3] >= max(8, n // 85)
        ]
        large_runs = sorted(large_runs, key=lambda run: (-run[3], run[1]))[:22]
        for clade, y0, y1, _count in sorted(large_runs, key=lambda run: run[1]):
            label_y = (y0 + y1) / 2
            if any(abs(label_y - prev) < 11 for prev in labeled_y):
                continue
            labeled_y.append(label_y)
            ax.plot([bracket_x, bracket_x], [y0 - 0.4, y1 + 0.4],
                    color="#111111", lw=0.45, solid_capstyle="butt",
                    clip_on=False, zorder=8)
            ax.plot([bracket_x - bracket_tick, bracket_x], [y0 - 0.4, y0 - 0.4],
                    color="#111111", lw=0.45, solid_capstyle="butt",
                    clip_on=False, zorder=8)
            ax.plot([bracket_x - bracket_tick, bracket_x], [y1 + 0.4, y1 + 0.4],
                    color="#111111", lw=0.45, solid_capstyle="butt",
                    clip_on=False, zorder=8)
            ax.text(bracket_x + bracket_tick * 0.45, label_y, clade,
                    va="center", ha="left", fontsize=4.7,
                    color="#111111", clip_on=False, zorder=9)
    ax.set_yticks([])
    ax.margins(x=0.02, y=0.02)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cfd7de")
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="x", colors="#5c6773", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel, fontsize=10)
    if figtree_style:
        axis_left, axis_right = ax.get_xlim()
        tick_start = int(math.ceil(axis_left))
        tick_end = int(math.floor(axis_right))
        if tick_end >= tick_start:
            tick_span = tick_end - tick_start
            tick_step = 1 if tick_span <= 18 else (2 if tick_span <= 32 else 5)
            ax.set_xticks(list(range(tick_start, tick_end + 1, tick_step)))
        ax.tick_params(axis="x", colors="#1f1f1f", labelsize=5.5, length=2, width=0.45)
        ax.spines["bottom"].set_color("#1f1f1f")
        ax.spines["bottom"].set_linewidth(0.45)
        ax.set_xlabel("")
    else:
        if display_note:
            title = f"{title}\n{display_note}"
        ax.set_title(title, fontsize=12, pad=8)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="none",
                          markerfacecolor=cmap[c], markeredgecolor="#f8f9fb",
                          markersize=8, label=c) for c in clades]
    if targets:
        handles.append(
            plt.Line2D([0], [0], marker="*", linestyle="none",
                       markerfacecolor="#ffd166", markeredgecolor="#111827",
                       markersize=12, label="target sequence")
        )
    if not figtree_style:
        legend_cols = 2 if len(handles) > 26 else 1
        legend_x = 1.12 if show_clade_bar else 1.005
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(legend_x, 1.0),
                  ncol=legend_cols, fontsize=7,
                  title="Nextclade clade", title_fontsize=8,
                  frameon=True, fancybox=False, framealpha=0.94,
                  borderpad=0.45, labelspacing=0.34,
                  columnspacing=0.85, handletextpad=0.42)

    if figtree_style:
        if focus_insets_drawn:
            ax.set_position([0.50, 0.035, 0.46, 0.945])
        else:
            plt.subplots_adjust(left=0.01, right=0.985, top=0.995, bottom=0.035)
    else:
        right_margin = 0.72 if show_clade_bar else 0.82
        plt.subplots_adjust(left=0.03, right=right_margin, top=0.93, bottom=0.08)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return {
        "tree_display_original_tips": original_tip_count,
        "tree_display_tips": n,
        "tree_display_max_tips": display_max_tips if figtree_style else 0,
        "tree_display_branch_cap_years": (
            display_branch_cap_years if figtree_style and x_by_name else 0
        ),
        "tree_display_sampled": bool(figtree_style and display_max_tips and n < original_tip_count),
    }


def find_external_executable(
    explicit: Optional[str],
    names: List[str],
    base: Path,
) -> Optional[str]:
    if explicit:
        found = shutil.which(explicit)
        if found:
            return found
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return str(explicit_path)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    candidates: List[Path] = []
    for root in (Path.cwd(), base, base.parent):
        for name in names:
            candidates.append(root / ".tools" / name)
            if not name.lower().endswith(".exe"):
                candidates.append(root / ".tools" / f"{name}.exe")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def run_logged_command(cmd: List[str], log_path: Path) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND:\n" + " ".join(cmd) + "\n\n"
        f"EXIT_CODE:\n{result.returncode}\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"external command failed (exit={result.returncode}): {' '.join(cmd)}\n"
            f"See log: {log_path}"
        )


def prepare_iqtree_alignment(
    tree_input_aa: Dict[str, str],
    raw_records: Dict[str, str],
    ref_prot: str,
    aligner: "PairwiseAligner",
    outdir: Path,
    prefer_nucleotide: bool = True,
) -> Tuple[Path, str, Dict[str, str], List[str]]:
    nt_alignment: Dict[str, str] = {}
    skipped_nt: List[str] = []
    if prefer_nucleotide:
        for name in tree_input_aa:
            raw = raw_records.get(name, "")
            projected_nt = project_coding_nt_to_reference(raw, ref_prot, aligner) if raw else ""
            if projected_nt:
                nt_alignment[name] = projected_nt
            else:
                skipped_nt.append(name)

    if prefer_nucleotide and len(nt_alignment) >= 3:
        aln_path = outdir / "tree_alignment_codon.fasta"
        write_fasta(aln_path, nt_alignment.items())
        return aln_path, "nucleotide_codon_projected", nt_alignment, skipped_nt

    aln_path = outdir / "tree_alignment_protein.fasta"
    write_fasta(aln_path, tree_input_aa.items())
    return aln_path, "protein_reference_projected", tree_input_aa, skipped_nt


def write_tree_dates_csv(
    names: Iterable[str],
    out_csv: Path,
    date_overrides: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    overrides = date_overrides or {}
    rows: List[Dict[str, str]] = []
    for name in names:
        date_value = find_date_override(overrides, name) or extract_collection_date(name)
        if date_value:
            rows.append({"name": name, "date": date_value})
    write_csv(out_csv, rows, ["name", "date"])
    return rows


def load_treetime_dates(path: Path) -> Dict[str, float]:
    dates: Dict[str, float] = {}
    if not path.exists():
        return dates
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 3:
                continue
            try:
                dates[row[0]] = float(row[2])
            except ValueError:
                continue
    return dates


def load_treetime_outliers(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row or not row[0].strip():
                continue
            item: Dict[str, object] = {
                "sample": row[0].strip(),
                "given_date": row[1].strip() if len(row) > 1 else "",
                "apparent_date": row[2].strip() if len(row) > 2 else "",
                "residual": row[3].strip() if len(row) > 3 else "",
                "reason": "treetime_temporal_outlier",
            }
            rows.append(item)
    return rows


def find_treetime_tree(tt_dir: Path) -> Tuple[Optional[Path], str]:
    preferred = [
        ("timetree.nexus", "nexus"),
        ("timetree.nwk", "newick"),
        ("timetree.newick", "newick"),
        ("annotated_tree.nexus", "nexus"),
        ("annotated_tree.nwk", "newick"),
    ]
    for filename, fmt in preferred:
        path = tt_dir / filename
        if path.exists():
            return path, fmt
    for path in tt_dir.glob("*.nexus"):
        return path, "nexus"
    for path in list(tt_dir.glob("*.nwk")) + list(tt_dir.glob("*.newick")):
        return path, "newick"
    return None, ""


def build_iqtree_treetime_outputs(
    tree_input: Dict[str, str],
    raw_records: Dict[str, str],
    ref_prot: str,
    aligner: "PairwiseAligner",
    clade_by_name: Dict[str, str],
    outdir: Path,
    base: Path,
    target_names: Iterable[str],
    vaccine_names: Iterable[str],
    display_note: str,
    iqtree_exe: Optional[str] = None,
    treetime_exe: Optional[str] = None,
    iqtree_model: str = "MFP",
    iqtree_threads: str = "AUTO",
    iqtree_fast: bool = False,
    run_treetime: bool = False,
    target_date: Optional[str] = None,
    tree_date_metadata: Optional[Dict[str, str]] = None,
    tree_date_metadata_path: str = "",
    tree_date_metadata_rows: int = 0,
    remove_treetime_outliers: bool = False,
    protected_names: Optional[Iterable[str]] = None,
    plot_style: str = "figtree",
    show_clade_bar: bool = False,
    display_max_tips: Optional[int] = None,
    display_branch_cap_years: Optional[float] = None,
    treetime_outlier_pass: int = 1,
    treetime_outlier_max_passes: int = 3,
    accumulated_treetime_outliers: Optional[List[Dict[str, object]]] = None,
    accumulated_metadata_outliers: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    workdir = outdir / "iqtree_treetime"
    workdir.mkdir(parents=True, exist_ok=True)
    stale_time_png = outdir / "time_scaled_tree.png"
    if stale_time_png.exists():
        stale_time_png.unlink()

    date_overrides = dict(tree_date_metadata or {})
    if target_date:
        for name in target_names:
            set_date_override(date_overrides, name, target_date)
    metadata_outliers = list(accumulated_metadata_outliers or [])
    tree_input_for_run = dict(tree_input)
    if run_treetime:
        newly_removed_metadata = find_temporal_metadata_conflicts(
            tree_input_for_run.keys(),
            protected_names or [],
            date_overrides=date_overrides,
        )
        if newly_removed_metadata:
            bad_names = {str(row["sample"]) for row in newly_removed_metadata}
            tree_input_for_run = {
                name: seq for name, seq in tree_input_for_run.items()
                if name not in bad_names
            }
            metadata_outliers.extend(newly_removed_metadata)
            metadata_csv = outdir / "tree_metadata_outliers_removed.csv"
            write_csv(
                metadata_csv,
                metadata_outliers,
                [
                    "sample",
                    "collection_date",
                    "collection_year",
                    "strain_year",
                    "year_difference",
                    "reason",
                ],
            )
            log(
                f"TreeTime 전에 날짜 metadata 충돌 {len(newly_removed_metadata)}개를 "
                "tree 입력에서 제거했습니다."
            )

    aln_path, aln_type, aligned_records, skipped_nt = prepare_iqtree_alignment(
        tree_input_for_run, raw_records, ref_prot, aligner, workdir, prefer_nucleotide=True
    )
    if skipped_nt and aln_type.startswith("nucleotide"):
        log(f"IQ-TREE nucleotide alignment에서 {len(skipped_nt)}개 서열은 codon projection 실패로 제외했습니다.")

    iqtree_cli = find_external_executable(iqtree_exe, ["iqtree2", "iqtree"], base)
    if not iqtree_cli:
        raise FileNotFoundError(
            "IQ-TREE executable not found. Install iqtree2 or pass --iqtree-exe <path>."
        )

    prefix = workdir / "ha_iqtree"
    cmd = [
        iqtree_cli,
        "-s", str(aln_path),
        "-m", iqtree_model,
        "-nt", str(iqtree_threads),
        "-pre", str(prefix),
        "-redo",
    ]
    if iqtree_fast:
        cmd.append("-fast")
    cmd.append("-quiet")
    log("IQ-TREE 실행: " + " ".join(cmd))
    run_logged_command(cmd, workdir / "iqtree.log")

    iqtree_tree = prefix.with_suffix(".treefile")
    if not iqtree_tree.exists():
        raise FileNotFoundError(f"IQ-TREE treefile not found: {iqtree_tree}")

    out_newick = outdir / "phylogenetic_tree.newick"
    shutil.copyfile(iqtree_tree, out_newick)
    render_stats: Dict[str, object] = {}
    if not run_treetime:
        render_stats = render_newick_tree_png(
            iqtree_tree,
            "newick",
            clade_by_name,
            outdir / "phylogenetic_tree.png",
            target_names=target_names,
            vaccine_names=vaccine_names,
            title="Phylogenetic tree - IQ-TREE maximum-likelihood",
            xlabel="substitutions per site",
            display_note=f"{display_note}; alignment={aln_type}",
            plot_style=plot_style,
            show_clade_bar=show_clade_bar,
            display_max_tips=display_max_tips,
            display_branch_cap_years=display_branch_cap_years,
        )

    dates_path = outdir / "tree_dates.csv"
    date_rows = write_tree_dates_csv(aligned_records.keys(), dates_path, date_overrides)
    tree_date_metadata_matched = sum(
        1 for name in aligned_records.keys()
        if find_date_override(tree_date_metadata, name)
    )

    outputs: Dict[str, object] = {
        "image_names": ["phylogenetic_tree.png"] if not run_treetime else [],
        "alignment": str(aln_path),
        "alignment_type": aln_type,
        "iqtree_workdir": str(workdir),
        "iqtree_treefile": str(iqtree_tree),
        "tree_dates": str(dates_path),
        "tree_input_sequences": len(tree_input_for_run),
        "tree_temporal_dates": len(date_rows),
        "tree_date_metadata": tree_date_metadata_path,
        "tree_date_metadata_rows": tree_date_metadata_rows,
        "tree_date_metadata_matched": tree_date_metadata_matched,
        "tree_alignment_sequences": len(aligned_records),
        "tree_nt_projection_skipped": len(skipped_nt) if aln_type.startswith("nucleotide") else 0,
        "tree_metadata_outliers_removed": len(metadata_outliers),
    }
    if metadata_outliers:
        outputs["tree_metadata_outliers_removed_csv"] = str(outdir / "tree_metadata_outliers_removed.csv")
    outputs.update(render_stats)

    if run_treetime:
        if not aln_type.startswith("nucleotide"):
            log("TreeTime은 nucleotide/codon alignment가 필요해서 protein alignment에서는 건너뜁니다.")
            outputs["treetime_status"] = "skipped_non_nucleotide_alignment"
            return outputs
        if len(date_rows) < 3:
            log("TreeTime 날짜가 3개 미만이라 timetree를 건너뜁니다. header 날짜 또는 --target-date를 확인하세요.")
            outputs["treetime_status"] = "skipped_insufficient_dates"
            return outputs

        treetime_cli = find_external_executable(treetime_exe, ["treetime"], base)
        if not treetime_cli:
            raise FileNotFoundError(
                "TreeTime executable not found. Install treetime or pass --treetime-exe <path>."
            )

        tt_dir = workdir / "treetime"
        if tt_dir.exists():
            shutil.rmtree(tt_dir)
        tt_cmd = [
            treetime_cli,
            "--aln", str(aln_path),
            "--tree", str(iqtree_tree),
            "--dates", str(dates_path),
            "--outdir", str(tt_dir),
            "--reroot", "least-squares",
        ]
        log("TreeTime 실행: " + " ".join(tt_cmd))
        treetime_log = workdir / "treetime.log"
        try:
            run_logged_command(tt_cmd, treetime_log)
        except RuntimeError:
            fail_text = treetime_log.read_text(encoding="utf-8", errors="ignore")
            if "Rerooting failed" not in fail_text and "No valid root found" not in fail_text:
                raise
            fallback_cmd = [
                treetime_cli,
                "--aln", str(aln_path),
                "--tree", str(iqtree_tree),
                "--dates", str(dates_path),
                "--outdir", str(tt_dir),
            ]
            log("TreeTime least-squares reroot failed; retrying without --reroot.")
            if tt_dir.exists():
                shutil.rmtree(tt_dir)
            run_logged_command(fallback_cmd, workdir / "treetime_fallback.log")
            outputs["treetime_reroot_fallback"] = "no_reroot"

        tt_tree, tt_format = find_treetime_tree(tt_dir)
        outputs["treetime_outdir"] = str(tt_dir)
        outputs["treetime_status"] = "completed"
        if remove_treetime_outliers:
            protected_keys = {tree_label_key(name).lower() for name in (protected_names or [])}
            protected_keys.update(normalize_id(name).lower() for name in (protected_names or []))
            treetime_outliers = load_treetime_outliers(tt_dir / "outliers.tsv")
            accumulated_rows = list(accumulated_treetime_outliers or [])
            removable_outliers: List[Dict[str, object]] = []
            filtered_tree_input = dict(tree_input_for_run)
            for row in treetime_outliers:
                name = str(row.get("sample", ""))
                name_keys = {tree_label_key(name).lower(), normalize_id(name).lower()}
                if name in filtered_tree_input and not (name_keys & protected_keys):
                    annotated_row = dict(row)
                    annotated_row["pass"] = treetime_outlier_pass
                    removable_outliers.append(annotated_row)
                    filtered_tree_input.pop(name, None)

            outlier_csv = outdir / "treetime_outliers_removed.csv"
            can_remove_this_pass = (
                treetime_outlier_pass <= treetime_outlier_max_passes
                and len(filtered_tree_input) >= 3
            )
            rows_to_write = (
                accumulated_rows + removable_outliers
                if can_remove_this_pass else accumulated_rows
            )
            write_csv(
                outlier_csv,
                rows_to_write,
                ["pass", "sample", "given_date", "apparent_date", "residual", "reason"],
            )
            outputs["treetime_outliers_detected"] = len(treetime_outliers)
            outputs["treetime_outliers_removed"] = len(rows_to_write)
            outputs["treetime_outliers_removed_this_pass"] = (
                len(removable_outliers) if can_remove_this_pass else 0
            )
            outputs["treetime_outlier_pass"] = treetime_outlier_pass
            outputs["treetime_outliers_removed_csv"] = str(outlier_csv)
            if (
                removable_outliers
                and can_remove_this_pass
            ):
                log(
                    f"TreeTime temporal outlier 제거 pass {treetime_outlier_pass}/"
                    f"{treetime_outlier_max_passes}: {len(removable_outliers)}개 제거 후 "
                    "IQ-TREE/TreeTime을 다시 실행합니다."
                )
                final_outputs = build_iqtree_treetime_outputs(
                    tree_input=filtered_tree_input,
                    raw_records=raw_records,
                    ref_prot=ref_prot,
                    aligner=aligner,
                    clade_by_name=clade_by_name,
                    outdir=outdir,
                    base=base,
                    target_names=target_names,
                    vaccine_names=vaccine_names,
                    display_note=f"{display_note}; {len(removable_outliers)} TreeTime temporal outliers removed",
                    iqtree_exe=iqtree_exe,
                    treetime_exe=treetime_exe,
                    iqtree_model=iqtree_model,
                    iqtree_threads=iqtree_threads,
                    iqtree_fast=iqtree_fast,
                    run_treetime=run_treetime,
                    target_date=target_date,
                    tree_date_metadata=tree_date_metadata,
                    tree_date_metadata_path=tree_date_metadata_path,
                    tree_date_metadata_rows=tree_date_metadata_rows,
                    remove_treetime_outliers=True,
                    protected_names=protected_names,
                    plot_style=plot_style,
                    show_clade_bar=show_clade_bar,
                    display_max_tips=display_max_tips,
                    display_branch_cap_years=display_branch_cap_years,
                    treetime_outlier_pass=treetime_outlier_pass + 1,
                    treetime_outlier_max_passes=treetime_outlier_max_passes,
                    accumulated_treetime_outliers=rows_to_write,
                    accumulated_metadata_outliers=metadata_outliers,
                )
                if treetime_outlier_pass == 1:
                    final_outputs["treetime_outliers_detected_first_pass"] = len(treetime_outliers)
                final_outputs["treetime_outliers_removed"] = int(
                    final_outputs.get("treetime_outliers_removed", len(rows_to_write)) or 0
                )
                final_outputs["treetime_outliers_removed_csv"] = str(outlier_csv)
                return final_outputs
            elif treetime_outliers:
                if removable_outliers and treetime_outlier_pass > treetime_outlier_max_passes:
                    log(
                        "TreeTime temporal outlier가 아직 남아 있지만 최대 반복 횟수에 도달해 "
                        "추가 재실행은 하지 않습니다."
                    )
                else:
                    log("TreeTime temporal outlier가 감지됐지만 보호 대상이거나 제거 후 서열 수가 부족해 재실행하지 않았습니다.")

        if tt_tree:
            outputs["treetime_tree"] = str(tt_tree)
            try:
                tt_dates = load_treetime_dates(tt_dir / "dates.tsv")
                tip_dates = [
                    tt_dates[name]
                    for name in aligned_records
                    if name in tt_dates and math.isfinite(tt_dates[name])
                ]
                node_dates = [
                    value
                    for name, value in tt_dates.items()
                    if str(name).startswith("NODE_") and math.isfinite(value)
                ]
                date_xlim: Optional[Tuple[float, float]] = None
                if tip_dates:
                    all_dates = tip_dates + node_dates
                    left = max(math.floor((min(all_dates) - 1.0) / 5.0) * 5.0, 1800.0)
                    right = max(tip_dates) + 1.0
                    if right <= left:
                        right = left + 1.0
                    date_xlim = (left, right)
                time_note = f"{display_note}; {len(date_rows)} dated tips used"
                if date_xlim:
                    time_note += f"; calendar axis {date_xlim[0]:.0f}-{date_xlim[1]:.0f}"
                render_stats = render_newick_tree_png(
                    tt_tree,
                    tt_format,
                    clade_by_name,
                    outdir / "phylogenetic_tree.png",
                    target_names=target_names,
                    vaccine_names=vaccine_names,
                    title="Time-scaled phylogeny - IQ-TREE + TreeTime",
                    xlabel="calendar year",
                    display_note=time_note,
                    x_by_name=tt_dates,
                    xlim=date_xlim,
                    plot_style=plot_style,
                    show_clade_bar=show_clade_bar,
                    display_max_tips=display_max_tips,
                    display_branch_cap_years=display_branch_cap_years,
                )
                outputs.update(render_stats)
                outputs["image_names"] = ["phylogenetic_tree.png"]
                outputs["time_scaled_tree"] = str(outdir / "phylogenetic_tree.png")
            except Exception as exc:
                log(f"TreeTime tree PNG 렌더링 실패: {exc}")
                outputs["treetime_render_error"] = str(exc)
        else:
            log("TreeTime 실행은 끝났지만 timetree 파일을 찾지 못했습니다.")
            outputs["treetime_status"] = "completed_no_tree_found"

    return outputs


def write_csv(path: Path, rows: List[Dict[str, object]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(
    out_html: Path,
    sanity_rows: List[Dict[str, object]],
    clade_rows: List[Dict[str, object]],
    antigenic_rows: List[Dict[str, object]],
    vaccine_rows: List[Dict[str, object]],
    drug_rows: List[Dict[str, object]],
    images: List[str],
) -> None:
    def table(rows: List[Dict[str, object]], cols: List[str]) -> str:
        if not rows:
            return "<p class='muted'>해당 항목 없음</p>"
        head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
        body = ""
        for r in rows:
            body += "<tr>" + "".join(
                f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    imgs = "".join(
        f"<h2>{html.escape(Path(p).stem)}</h2><img src='{html.escape(p)}'>"
        for p in images
    )
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>H3N2 HA 분석 결과</title>
<style>
 body{{font-family:'Malgun Gothic',Arial,sans-serif;margin:24px;color:#1f2933}}
 h1{{font-size:24px}} h2{{font-size:18px;margin-top:28px}}
 table{{border-collapse:collapse;margin:8px 0;font-size:13px}}
 th,td{{border:1px solid #d9e2ec;padding:5px 9px;text-align:left}}
 th{{background:#f5f7fa}} img{{max-width:100%;border:1px solid #d9e2ec;border-radius:6px}}
 .muted{{color:#888}} .note{{color:#52616b;max-width:900px;line-height:1.5}}
</style></head><body>
<h1>H3N2 HA 분석 결과</h1>
<p class="note">관찰된 HA 서열에 대한 후향적 분석입니다(알려진 항원부위/약제부위 주석, 거리 계산, 계통수).
실험 지침이나 변이 설계가 아닙니다.</p>
{imgs}
<h2>H3 넘버링 점검 (reference)</h2>
<p class="note">아래 위치의 reference 잔기가 알려진 값과 맞는지 확인하세요. 어긋나면 상단 H3_OFFSET 을 조정하세요.</p>
{table(sanity_rows, ['h3_position','antigenic_site','reference_aa'])}
<h2>클레이드 지정</h2>
{table(clade_rows, ['sample','assigned_clade','source','qc_status','nextclade_error','score'])}
<h2>항원부위 변이 (reference 대비)</h2>
{table(antigenic_rows, ['sample','antigenic_site','h3_position','mutation','weight'])}
<h2>백신주 대비 항원거리</h2>
<p class="note">각 target 이 백신주(vaccine)에서 항원적으로 얼마나 떨어졌는지. differing_sites 는 차이가 난 항원부위(백신주잔기·위치·샘플잔기).</p>
{table(vaccine_rows, ['sample','vaccine','antigenic_distance','antigenic_differences','sites_compared','differing_sites'])}
<h2>약제 작용부위</h2>
{table(drug_rows, ['sample','drug','h3_position','mutation','changed','note'])}
</body></html>"""
    out_html.write_text(doc, encoding="utf-8")


# ==============================================================================
# ===  메인 파이프라인  =========================================================
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="H3N2 HA 항원·약제·계통 분석")
    parser.add_argument("--target", default=TARGET_FASTA)
    parser.add_argument("--reference", default=REFERENCE_FASTA)
    parser.add_argument("--background", default=BACKGROUND_FASTA)
    parser.add_argument("--vaccine", default=VACCINE_FASTA)
    parser.add_argument("--outdir", default=OUTPUT_DIR)
    parser.add_argument("--h3-offset", type=int, default=None,
                        help="H3 넘버링 보정값(기본: 코드 상단 H3_OFFSET)")
    parser.add_argument("--min-identity", type=float, default=LOW_IDENTITY_WARN_PERCENT,
                        help="reference projection identity warning threshold")
    parser.add_argument("--max-tree-sequences", type=int, default=MAX_TREE_SEQUENCES,
                        help="maximum sequences used for tree rendering; 0 means no limit")
    parser.add_argument(
        "--tree-method", default="auto",
        choices=("auto", "nj", "fast-upgma", "iqtree", "iqtree-treetime"),
        help="tree backend: auto/NJ/fast-upgma for dashboard tree, iqtree or iqtree-treetime for ML/time tree"
    )
    parser.add_argument("--iqtree-exe", default=None,
                        help="optional path/name for IQ-TREE executable, e.g. iqtree2.exe")
    parser.add_argument("--iqtree-model", default="MFP",
                        help="IQ-TREE model option, default MFP")
    parser.add_argument("--iqtree-threads", default="AUTO",
                        help="IQ-TREE -nt value, default AUTO")
    parser.add_argument("--iqtree-fast", action="store_true",
                        help="use IQ-TREE -fast mode for quicker full-background preview trees")
    parser.add_argument("--treetime-exe", default=None,
                        help="optional path/name for TreeTime executable")
    parser.add_argument("--target-date", default=None,
                        help="collection date for target sequence if not present in FASTA id, e.g. 2025-01-20")
    parser.add_argument("--tree-date-metadata", default=None,
                        help="CSV/TSV with sequence name and collection date columns for TreeTime, e.g. name,date")
    parser.add_argument("--tree-plot-style", default="figtree",
                        choices=("figtree", "dashboard"),
                        help="tree PNG style: figtree=clean presentation style, dashboard=colored clade legend")
    parser.add_argument("--tree-display-max-tips", type=int, default=0,
                        help="max tips drawn in figtree PNG; 0 keeps every tip in the static image")
    parser.add_argument("--tree-display-branch-cap", type=float, default=0.65,
                        help="max displayed horizontal branch length in years for figtree PNG; 0 disables visual compression")
    parser.add_argument("--tree-clade-bar", action="store_true",
                        help="add right-side clade segment bar in dashboard-style tree rendering")
    parser.add_argument("--treetime-remove-outliers", action="store_true",
                        help="iteratively remove tips in TreeTime outliers.tsv and rerun IQ-TREE/TreeTime")
    parser.add_argument("--treetime-outlier-max-passes", type=int, default=3,
                        help="maximum TreeTime temporal outlier removal passes, default 3")
    parser.add_argument("--tree-outliers", default="",
                        help="comma/semicolon-separated tip names to remove from tree input")
    parser.add_argument("--tree-outlier-file", default=None,
                        help="optional text file with one tree outlier tip per line")
    parser.add_argument("--tree-date-min", type=int, default=None,
                        help="remove non-protected tree tips with collection year before this value")
    parser.add_argument("--tree-date-max", type=int, default=None,
                        help="remove non-protected tree tips with collection year after this value")
    parser.add_argument(
        "--clade-method", default="auto",
        choices=("auto", "rules", "nextclade"),
        help="클레이드 판정 방법: auto=nextclade 우선, rules=예시 규칙, nextclade=실제 nextclade 호출"
    )
    parser.add_argument(
        "--nextclade-dataset", default=None,
        help="Nextclade dataset 경로/이름(지정하지 않으면 CLI 기본 설정 사용)"
    )
    parser.add_argument(
        "--nextclade-results", default=None,
        help="Nextclade Web/CLI에서 내려받은 CSV/TSV/JSON 결과 파일. 지정하면 이 파일의 clade를 최우선 사용"
    )
    parser.add_argument(
        "--allow-rule-clade-fallback", action="store_true",
        help="auto 모드에서 Nextclade를 사용할 수 없을 때 예시 marker rule로 임시 clade를 붙임"
    )
    args = parser.parse_args(argv)

    global H3_OFFSET
    if args.h3_offset is not None:
        H3_OFFSET = args.h3_offset

    base = Path(__file__).resolve().parent
    outdir = resolve_input_path(base, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 입력 로드 -------------------------------------------------------------
    ref_path = resolve_input_path(base, args.reference)
    tgt_path = resolve_input_path(base, args.target)
    bg_path = resolve_input_path(base, args.background)
    vac_path = resolve_input_path(base, args.vaccine)
    tree_date_metadata: Dict[str, str] = {}
    tree_date_metadata_rows = 0
    tree_date_metadata_path = ""
    if args.tree_date_metadata:
        try:
            metadata_path = resolve_input_path(base, args.tree_date_metadata)
            tree_date_metadata, tree_date_metadata_rows = load_tree_date_metadata(metadata_path)
            tree_date_metadata_path = str(metadata_path)
            log(
                f"TreeTime date metadata loaded: {tree_date_metadata_rows} rows "
                f"from {metadata_path}"
            )
        except Exception as exc:
            log(f"TreeTime date metadata read failed: {exc}")
            return 2
    for label, p in [("reference", ref_path), ("target", tgt_path)]:
        if not p.exists():
            log(f"[필수 파일 없음] {label}: {p.name} 를 이 폴더에 넣어주세요.")
            return 2

    ref_records = read_fasta(ref_path)
    if len(ref_records) > 1:
        log(f"reference 에 서열이 여러 개라 첫 번째({ref_records[0][0]})를 사용합니다.")
    ref_id, ref_raw = ref_records[0]
    ref_prot = to_protein(ref_raw)
    log(f"reference: {ref_id} ({len(ref_prot)} aa)")

    # --- reference 좌표로 번역+정렬(투영) -------------------------------------
    # 핵산이면 6프레임 중 reference 에 가장 잘 붙는 것을 자동 선택(방향/프레임 자동, UTR 자동 절단).
    aligner = make_aligner()
    ref_proj = ref_prot  # reference 자기 자신은 좌표 그대로
    low_identity: List[str] = []

    def project(name: str, raw: str) -> Tuple[str, str]:
        prot = to_protein_vs_reference(raw, ref_prot, aligner)
        proj = align_to_reference(ref_prot, prot, aligner)
        ident = projection_identity(ref_proj, proj)
        kind = "핵산→번역" if is_nucleotide(re.sub(r"[^A-Z*]", "", raw.upper())) else "단백질"
        log(f"  - {name}: {kind}, reference 일치도 {ident}%")
        if ident < args.min_identity:
            low_identity.append(f"{name}({ident}%)")
        return name, proj

    proj_targets = dict(project(normalize_id(n), s) for n, s in read_fasta(tgt_path))
    proj_background = {}
    if bg_path.exists():
        proj_background = dict(project(normalize_id(n), s) for n, s in read_fasta(bg_path))
    else:
        log(f"background({bg_path.name}) 없음 → 계통수는 target 만으로 그립니다.")
    # vaccine(백신주): cartography·항원거리 비교 기준. 없으면 reference 를 대용.
    if vac_path.exists():
        proj_vaccine = dict(project(normalize_id(n), s) for n, s in read_fasta(vac_path))
    else:
        proj_vaccine = {normalize_id(ref_id): ref_proj}
        log(f"vaccine({vac_path.name}) 없음 → reference 를 백신주 대용으로 사용합니다.")
    log(f"정렬 완료: target {len(proj_targets)}개, vaccine {len(proj_vaccine)}개, "
        f"background {len(proj_background)}개")
    if low_identity:
        log(
            f"⚠️ reference 일치도가 {args.min_identity}% 미만인 서열"
            "(HA·서브타입·품질 확인 필요): " + ", ".join(low_identity)
        )

    # --- H3 넘버링 점검표(사용자가 OFFSET 검증용) ----------------------------
    sanity_rows = []
    for site in [145, 155, 156, 158, 159, 189, 193]:
        label, _ = ANTIGENIC_SITES.get(site, ("-", 0))
        sanity_rows.append({"h3_position": site, "antigenic_site": label,
                            "reference_aa": residue_at(ref_proj, site) or "(좌표밖)"})
    log("H3 넘버링 점검(reference 잔기): " +
        ", ".join(f"{r['h3_position']}={r['reference_aa']}" for r in sanity_rows))

    # --- 실제 클레이드 호출 시도 ----------------------------------------------
    clade_by_name: Dict[str, str] = {}
    clade_qc_by_name: Dict[str, str] = {}
    clade_error_by_name: Dict[str, str] = {}
    clade_source = "none"
    raw_targets = dict((normalize_id(n), s) for n, s in read_fasta(tgt_path))
    raw_background = dict((normalize_id(n), s) for n, s in read_fasta(bg_path)) if bg_path.exists() else {}
    raw_vaccine = dict((normalize_id(n), s) for n, s in read_fasta(vac_path)) if vac_path.exists() else {}
    query_records = list(raw_targets.items()) + list(raw_background.items()) + list(raw_vaccine.items())
    if query_records:
        write_fasta(outdir / "nextclade_queries.fasta", query_records)

    if args.nextclade_results:
        nextclade_results_path = resolve_input_path(base, args.nextclade_results)
        try:
            clade_map = load_nextclade_clade_file(nextclade_results_path)
        except Exception as exc:
            log(f"Nextclade 결과 파일을 읽지 못했습니다: {exc}")
            return 2
        for name, meta in clade_map.items():
            clade_by_name[name] = meta.get("clade", "") or "unassigned"
            clade_qc_by_name[name] = meta.get("qc_status", "")
            clade_error_by_name[name] = meta.get("errors", "")
        clade_source = "nextclade_file"
        log(f"Nextclade 결과 파일 기준으로 클레이드를 지정했습니다: {nextclade_results_path}")
    elif args.clade_method in ("auto", "nextclade"):
        nextclade_exe = find_nextclade_executable(base)
        nextclade_available = nextclade_exe is not None
        if nextclade_available:
            if not args.nextclade_dataset:
                if args.clade_method == "nextclade" or not args.allow_rule_clade_fallback:
                    log("Nextclade CLI clade 판정에는 --nextclade-dataset 이 필요합니다.")
                    log("또는 Nextclade Web/CLI 결과 TSV/CSV/JSON을 --nextclade-results 로 넣으세요.")
                    return 2
                log("--nextclade-dataset 이 없어, 사용자가 허용한 예시 규칙 기반 clade로 진행합니다.")
            else:
                try:
                    if query_records:
                        clade_map = run_nextclade_clade_call(
                            query_records=query_records,
                            ref_path=ref_path,
                            out_json=outdir / "nextclade_results.json",
                            dataset=args.nextclade_dataset,
                            cli_path=nextclade_exe,
                        )
                        for name, meta in clade_map.items():
                            clade_by_name[name] = meta.get("clade", "") or "unassigned"
                            clade_qc_by_name[name] = meta.get("qc_status", "")
                            clade_error_by_name[name] = meta.get("errors", "")
                        clade_source = "nextclade_cli"
                        log("Nextclade CLI 결과로 클레이드를 지정했습니다.")
                except Exception as exc:
                    if args.clade_method == "nextclade" or not args.allow_rule_clade_fallback:
                        log(f"Nextclade 호출 실패: {exc}")
                        log("공식 clade가 필요하면 --nextclade-results 또는 --nextclade-dataset 을 지정하세요.")
                        return 2
                    log(f"Nextclade 호출 실패, 사용자가 허용한 예시 규칙 기반 clade로 진행합니다: {exc}")
        else:
            if args.clade_method == "nextclade" or not args.allow_rule_clade_fallback:
                log("nextclade 실행 파일을 찾지 못했습니다.")
                log("공식 clade가 필요하면 Nextclade Web/CLI 결과를 --nextclade-results 로 넣으세요.")
                return 2
            log("nextclade 실행 파일을 찾지 못해, 사용자가 허용한 예시 규칙 기반 clade로 진행합니다.")

    # --- 항원부위 변이 ---------------------------------------------------------
    antigenic_rows = antigenic_mutation_table(ref_proj, proj_targets)
    write_csv(outdir / "antigenic_site_mutations.csv", antigenic_rows,
              ["sample", "antigenic_site", "h3_position", "reference_aa",
               "observed_aa", "mutation", "weight"])

    # --- 약제 작용부위 변이 ----------------------------------------------------
    drug_rows = drug_mutation_table(ref_proj, proj_targets)
    write_csv(outdir / "drug_site_mutations.csv", drug_rows,
              ["sample", "drug", "h3_position", "reference_aa", "observed_aa",
               "mutation", "changed", "note"])
    log(DRUG_SITES_NOTE)

    # --- 클레이드 지정 ---------------------------------------------------------
    clade_rows = []
    if not clade_by_name:
        for name, proj in {**proj_targets, **proj_background, **proj_vaccine}.items():
            clade, score = assign_clade(proj)
            clade_by_name[name] = clade
            if name in proj_targets or name in proj_background:
                clade_rows.append({
                    "sample": name,
                    "assigned_clade": clade,
                    "source": "provisional_rule",
                    "qc_status": "",
                    "nextclade_error": "",
                    "score": score,
                })
        clade_by_name[normalize_id(ref_id)] = assign_clade(ref_proj)[0]
    else:
        # 실 클레이드 결과가 있으면 score 는 별도 계산하지 않음(실측값은 외부 도구 기준)
        for name in list(proj_targets) + list(proj_background):
            assigned = clade_by_name.get(name, "unassigned")
            source = clade_source if name in clade_by_name else "missing_nextclade_result"
            clade_rows.append({
                "sample": name,
                "assigned_clade": assigned,
                "source": source,
                "qc_status": clade_qc_by_name.get(name, ""),
                "nextclade_error": clade_error_by_name.get(name, ""),
                "score": "-",
            })
        # reference 는 별도 처리
        clade_by_name[normalize_id(ref_id)] = clade_by_name.get(normalize_id(ref_id), "unassigned")

    write_csv(outdir / "clade_assignments.csv", clade_rows,
              ["sample", "assigned_clade", "source", "qc_status", "nextclade_error", "score"])
    clade_source_counts: Dict[str, int] = {}
    for row in clade_rows:
        src = str(row.get("source", ""))
        clade_source_counts[src] = clade_source_counts.get(src, 0) + 1
    effective_clade_source = (
        clade_source if clade_source != "none"
        else ("provisional_rule" if clade_rows else "none")
    )

    # --- 백신주 대비 항원거리 (실무적으로 가장 중요) ---------------------------
    vaccine_rows = antigenic_distance_to_vaccine(proj_targets, proj_vaccine)
    write_csv(outdir / "antigenic_distance_to_vaccine.csv", vaccine_rows,
              ["sample", "vaccine", "antigenic_distance", "antigenic_differences",
               "sites_compared", "differing_sites"])

    # --- Antigenic cartography (target + vaccine 만 비교) -----------------------
    carto_names, carto_groups, carto_proj = [], [], []
    for n, p in proj_vaccine.items():
        carto_names.append(n)
        carto_groups.append("vaccine")
        carto_proj.append(p)
    for n, p in proj_targets.items():
        carto_names.append(n)
        carto_groups.append("target")
        carto_proj.append(p)
    dmat = [[antigenic_distance(a, b) for b in carto_proj] for a in carto_proj]
    coords = classical_mds(dmat)
    draw_cartography(carto_names, carto_groups, coords, clade_by_name,
                     outdir / "antigenic_cartography.png")
    write_csv(outdir / "antigenic_cartography_coords.csv",
              [{"sample": n, "group": g, "x": round(x, 5), "y": round(y, 5)}
               for n, g, (x, y) in zip(carto_names, carto_groups, coords)],
              ["sample", "group", "x", "y"])

    # --- 계통수 ----------------------------------------------------------------
    # 전체 배경 서열을 포함하되, 이름 라벨은 숨긴 상태로 그림.
    tree_input, skipped_tree_background = build_tree_input(
        ref_id=ref_id,
        ref_proj=ref_proj,
        targets=proj_targets,
        background=proj_background,
        vaccines=proj_vaccine,
        max_sequences=args.max_tree_sequences,
    )
    if skipped_tree_background:
        log(
            f"tree 시각화 속도를 위해 background {skipped_tree_background}개를 "
            f"균등 간격으로 제외했습니다. 전체 사용은 --max-tree-sequences 0 으로 설정하세요."
        )
    try:
        tree_outlier_terms = load_tree_outlier_terms(
            args.tree_outliers,
            args.tree_outlier_file,
            base,
        )
    except Exception as exc:
        log(f"tree outlier 목록을 읽지 못했습니다: {exc}")
        return 2
    protected_tree_names = (
        [normalize_id(ref_id)]
        + list(proj_targets.keys())
        + list(proj_vaccine.keys())
    )
    tree_input, tree_outlier_rows = filter_tree_outliers(
        tree_input=tree_input,
        clade_by_name=clade_by_name,
        outlier_terms=tree_outlier_terms,
        protected_names=protected_tree_names,
        out_csv=outdir / "tree_outliers_removed.csv",
        date_min=args.tree_date_min,
        date_max=args.tree_date_max,
    )
    if tree_outlier_rows:
        log(f"tree outlier {len(tree_outlier_rows)}개를 IQ-TREE/시각화 입력에서 제거했습니다.")

    images = ["antigenic_cartography.png"]
    effective_tree_method = ""
    tree_extra_outputs: Dict[str, object] = {}
    tree_alignment_type = ""
    tree_temporal_dates = 0
    tree_alignment_sequences = 0
    tree_nt_projection_skipped = 0
    treetime_outliers_removed = 0
    treetime_outliers_detected_final = 0
    treetime_outlier_pass = 0
    tree_metadata_outliers_removed = 0
    tree_date_metadata_matched = 0
    tree_display_original_tips = 0
    tree_display_tips = 0
    tree_display_sampled = False
    tree_input_sequences_final = len(tree_input)
    if len(tree_input) >= 3:
        effective_tree_method = args.tree_method
        if effective_tree_method == "auto":
            effective_tree_method = "fast-upgma" if len(tree_input) > 300 else "nj"
        tree_note = (
            f"{len(tree_input)} sequences shown; "
            f"{skipped_tree_background} background sequences hidden by --max-tree-sequences"
            if skipped_tree_background
            else f"{len(tree_input)} sequences shown; all background sequences included"
        )
        if tree_outlier_rows:
            tree_note += f"; {len(tree_outlier_rows)} tree outliers removed"
        if effective_tree_method in ("iqtree", "iqtree-treetime"):
            raw_tree_records = {normalize_id(ref_id): ref_raw}
            raw_tree_records.update(raw_targets)
            raw_tree_records.update(raw_vaccine)
            raw_tree_records.update(raw_background)
            try:
                tree_extra_outputs = build_iqtree_treetime_outputs(
                    tree_input=tree_input,
                    raw_records=raw_tree_records,
                    ref_prot=ref_prot,
                    aligner=aligner,
                    clade_by_name=clade_by_name,
                    outdir=outdir,
                    base=base,
                    target_names=proj_targets.keys(),
                    vaccine_names=proj_vaccine.keys(),
                    display_note=tree_note,
                    iqtree_exe=args.iqtree_exe,
                    treetime_exe=args.treetime_exe,
                    iqtree_model=args.iqtree_model,
                    iqtree_threads=args.iqtree_threads,
                    iqtree_fast=args.iqtree_fast,
                    run_treetime=(effective_tree_method == "iqtree-treetime"),
                    target_date=args.target_date,
                    tree_date_metadata=tree_date_metadata,
                    tree_date_metadata_path=tree_date_metadata_path,
                    tree_date_metadata_rows=tree_date_metadata_rows,
                    remove_treetime_outliers=(
                        args.treetime_remove_outliers
                        and effective_tree_method == "iqtree-treetime"
                    ),
                    treetime_outlier_max_passes=args.treetime_outlier_max_passes,
                    protected_names=protected_tree_names,
                    plot_style=args.tree_plot_style,
                    show_clade_bar=args.tree_clade_bar,
                    display_max_tips=args.tree_display_max_tips,
                    display_branch_cap_years=args.tree_display_branch_cap,
                )
            except Exception as exc:
                log(f"IQ-TREE/TreeTime tree 생성 실패: {exc}")
                return 2
            image_names = tree_extra_outputs.get("image_names", [])
            if isinstance(image_names, list):
                images.extend(str(p) for p in image_names)
            tree_alignment_type = str(tree_extra_outputs.get("alignment_type", ""))
            tree_temporal_dates = int(tree_extra_outputs.get("tree_temporal_dates", 0) or 0)
            tree_alignment_sequences = int(tree_extra_outputs.get("tree_alignment_sequences", 0) or 0)
            tree_nt_projection_skipped = int(tree_extra_outputs.get("tree_nt_projection_skipped", 0) or 0)
            treetime_outliers_removed = int(tree_extra_outputs.get("treetime_outliers_removed", 0) or 0)
            treetime_outliers_detected_final = int(
                tree_extra_outputs.get("treetime_outliers_detected", 0) or 0
            )
            treetime_outlier_pass = int(tree_extra_outputs.get("treetime_outlier_pass", 0) or 0)
            tree_metadata_outliers_removed = int(
                tree_extra_outputs.get("tree_metadata_outliers_removed", 0) or 0
            )
            tree_date_metadata_matched = int(
                tree_extra_outputs.get("tree_date_metadata_matched", 0) or 0
            )
            tree_display_original_tips = int(
                tree_extra_outputs.get("tree_display_original_tips", 0) or 0
            )
            tree_display_tips = int(tree_extra_outputs.get("tree_display_tips", 0) or 0)
            tree_display_sampled = bool(tree_extra_outputs.get("tree_display_sampled", False))
            tree_input_sequences_final = int(tree_extra_outputs.get("tree_input_sequences", len(tree_input)) or len(tree_input))
        elif effective_tree_method == "fast-upgma":
            build_fast_upgma_tree_png(
                tree_input, clade_by_name,
                outdir / "phylogenetic_tree.png",
                outdir / "phylogenetic_tree.newick",
                target_names=proj_targets.keys(),
                display_note=tree_note,
            )
        else:
            build_tree_png(
                tree_input, clade_by_name,
                outdir / "phylogenetic_tree.png",
                outdir / "phylogenetic_tree.newick",
                target_names=proj_targets.keys(),
                display_note=tree_note,
            )
        if effective_tree_method not in ("iqtree", "iqtree-treetime"):
            images.append("phylogenetic_tree.png")
    else:
        log("서열이 3개 미만이라 계통수는 건너뜁니다(NJ 트리는 최소 3개 필요).")

    # --- 요약 리포트 -----------------------------------------------------------
    write_report(outdir / "report.html", sanity_rows, clade_rows,
                 antigenic_rows, vaccine_rows, drug_rows, images)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "analysis_scope": ANALYSIS_SCOPE_NOTICE,
        "inputs": {
            "target": str(tgt_path),
            "reference": str(ref_path),
            "background": str(bg_path) if bg_path.exists() else "",
            "vaccine": str(vac_path) if vac_path.exists() else "",
        },
        "parameters": {
            "h3_offset": H3_OFFSET,
            "min_identity": args.min_identity,
            "max_tree_sequences": args.max_tree_sequences,
            "clade_method": args.clade_method,
            "nextclade_dataset": args.nextclade_dataset or "",
            "nextclade_results": args.nextclade_results or "",
            "allow_rule_clade_fallback": args.allow_rule_clade_fallback,
            "tree_method": args.tree_method,
            "tree_method_effective": effective_tree_method,
            "tree_plot_style": args.tree_plot_style,
            "tree_display_max_tips": args.tree_display_max_tips,
            "tree_display_branch_cap": args.tree_display_branch_cap,
            "tree_clade_bar": args.tree_clade_bar,
            "tree_alignment_type": tree_alignment_type,
            "iqtree_model": args.iqtree_model,
            "iqtree_threads": args.iqtree_threads,
            "iqtree_fast": args.iqtree_fast,
            "target_date": args.target_date or "",
            "tree_date_metadata": tree_date_metadata_path,
            "treetime_remove_outliers": args.treetime_remove_outliers,
            "treetime_outlier_max_passes": args.treetime_outlier_max_passes,
            "tree_outliers": args.tree_outliers or "",
            "tree_outlier_file": args.tree_outlier_file or "",
            "tree_date_min": args.tree_date_min,
            "tree_date_max": args.tree_date_max,
            "clade_assignment_source": effective_clade_source,
        },
        "counts": {
            "targets": len(proj_targets),
            "background": len(proj_background),
            "vaccines": len(proj_vaccine),
            "tree_sequences": tree_input_sequences_final,
            "tree_background_skipped": skipped_tree_background,
            "tree_outliers_removed": len(tree_outlier_rows),
            "tree_display_original_tips": tree_display_original_tips,
            "tree_display_tips": tree_display_tips,
            "tree_display_sampled": tree_display_sampled,
            "tree_alignment_sequences": tree_alignment_sequences,
            "tree_temporal_dates": tree_temporal_dates,
            "tree_date_metadata_rows": tree_date_metadata_rows,
            "tree_date_metadata_matched": tree_date_metadata_matched,
            "tree_nt_projection_skipped": tree_nt_projection_skipped,
            "tree_metadata_outliers_removed": tree_metadata_outliers_removed,
            "treetime_outlier_pass": treetime_outlier_pass,
            "treetime_outliers_detected_final": treetime_outliers_detected_final,
            "treetime_outliers_removed": treetime_outliers_removed,
            "clade_rows": len(clade_rows),
            "clade_source_counts": clade_source_counts,
            "antigenic_mutations": len(antigenic_rows),
            "drug_site_rows": len(drug_rows),
        },
        "outputs": {
            "report": str(outdir / "report.html"),
            "nextclade_query_fasta": str(outdir / "nextclade_queries.fasta"),
            "antigenic_site_mutations": str(outdir / "antigenic_site_mutations.csv"),
            "drug_site_mutations": str(outdir / "drug_site_mutations.csv"),
            "clade_assignments": str(outdir / "clade_assignments.csv"),
            "antigenic_distance_to_vaccine": str(outdir / "antigenic_distance_to_vaccine.csv"),
            "antigenic_cartography": str(outdir / "antigenic_cartography.png"),
            "phylogenetic_tree": str(outdir / "phylogenetic_tree.png"),
            "phylogenetic_tree_newick": str(outdir / "phylogenetic_tree.newick"),
            "tree_outliers_removed": str(outdir / "tree_outliers_removed.csv"),
            "tree_metadata_outliers_removed": str(tree_extra_outputs.get("tree_metadata_outliers_removed_csv", "")),
            "tree_alignment": str(tree_extra_outputs.get("alignment", "")),
            "tree_dates": str(tree_extra_outputs.get("tree_dates", "")),
            "iqtree_workdir": str(tree_extra_outputs.get("iqtree_workdir", "")),
            "iqtree_treefile": str(tree_extra_outputs.get("iqtree_treefile", "")),
            "treetime_outdir": str(tree_extra_outputs.get("treetime_outdir", "")),
            "treetime_tree": str(tree_extra_outputs.get("treetime_tree", "")),
            "treetime_outliers_removed": str(tree_extra_outputs.get("treetime_outliers_removed_csv", "")),
            "time_scaled_tree": str(tree_extra_outputs.get("time_scaled_tree", "")),
        },
    }
    (outdir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log(f"완료! 결과 폴더: {outdir}")
    log(f"요약 리포트를 브라우저로 여세요: {outdir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
