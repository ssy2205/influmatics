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
  - 터미널에서:   python h3n2_ha_analysis.py
  - 결과는 results/ 폴더에 표(CSV)와 그림(PNG), 요약(HTML)으로 저장된다.

필요한 패키지 (최초 1회):
        pip install biopython numpy matplotlib

설계 메모:
  - 입력이 핵산(DNA/RNA)이면 자동으로 단백질로 번역한다(3개 프레임 중 종결코돈이 가장 적은 것).
  - 모든 site 데이터(항원부위/약제부위/클레이드 규칙)는 단백질 H3 넘버링이다.
  - 이 도구는 "관찰된 서열에 알려진 표식을 주석/요약/시각화"하는 후향적 분석 도구다.
    새로운 변이를 설계하거나 적합도/회피를 예측하지 않는다.
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
OUTPUT_DIR = "results"

# --- H3 넘버링 보정값 -----------------------------------------------------------
# H3 넘버링은 "성숙한 HA1"의 첫 잔기를 1번으로 센다.
# reference 서열이 신호펩타이드(signal peptide, H3는 보통 16잔기)를 포함하면 16을 쓴다.
# reference 가 이미 성숙형(HA1 시작)이면 0 으로 바꾼다.
#   reference 의 (1-based) 잔기 번호 = H3site + H3_OFFSET
H3_OFFSET = 16

# --- 항원부위 (H3 넘버링) -------------------------------------------------------
# 출처: Wiley, Wilson & Skehel (1981) Nature; Skehel & Wiley (2000) Annu Rev Biochem.
#       Koel et al. (2013) Science 의 핵심 7개 위치는 가중치 2로 강조.
# 형식: { H3site: (항원부위라벨, 가중치) }
ANTIGENIC_SITES: Dict[int, Tuple[str, float]] = {}
def _add_sites(label: str, positions: List[int], weight: float = 1.0) -> None:
    for p in positions:
        # 이미 있으면 더 큰 가중치를 유지
        prev = ANTIGENIC_SITES.get(p)
        if prev is None or weight > prev[1]:
            ANTIGENIC_SITES[p] = (label, weight)

_add_sites("Site_A", [122, 124, 126, 131, 133, 135, 137, 142, 143, 144, 145, 146, 150, 152, 168])
_add_sites("Site_B", [128, 129, 155, 156, 157, 158, 159, 160, 163, 164, 165, 186, 187, 188, 189, 190, 192, 193, 196, 197, 198])
_add_sites("Site_C", [44, 45, 46, 48, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294, 297, 299, 300, 304, 305, 307, 308, 309, 310, 311, 312])
_add_sites("Site_D", [96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248])
_add_sites("Site_E", [57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265])
# Koel 2013 핵심 7개 위치 — 항원 진화에 가장 큰 영향(가중치 2)
_add_sites("Koel7", [145, 155, 156, 158, 159, 189, 193], weight=2.0)

# --- 항바이러스제 작용부위 (H3 넘버링) -----------------------------------------
# ⚠️ 중요: 고전적 NA 억제제(oseltamivir 등) 내성변이(H275Y 등)는 NA 유전자에 있으며 HA 에는 없다.
#          HA 를 표적하는 약제는 umifenovir(arbidol; HA 줄기/삼량체 계면 결합)가 대표적이다.
#          아래 목록은 "예시/뼈대"이며, 반드시 최신 문헌으로 검증·교체할 것.
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


def to_protein(seq: str) -> str:
    """핵산이면 단백질로 번역(3개 정방향 프레임 중 내부 종결코돈이 가장 적은 것). 단백질이면 그대로."""
    s = re.sub(r"[^A-Z]", "", seq.upper())
    if not is_nucleotide(s):
        return s  # 이미 단백질
    s = s.replace("U", "T")
    best: Optional[Tuple[int, str]] = None
    for frame in range(3):
        sub = s[frame:]
        sub = sub[: len(sub) // 3 * 3]
        if not sub:
            continue
        prot = str(Seq(sub).translate())
        internal_stops = prot.rstrip("*").count("*")
        if best is None or internal_stops < best[0]:
            best = (internal_stops, prot)
    prot = (best[1] if best else "").rstrip("*")
    return prot


def normalize_id(name: str) -> str:
    name = re.sub(r"[\s/|:;,()\[\]']+", "_", str(name).strip())
    return re.sub(r"_+", "_", name).strip("_") or "seq"


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


def draw_cartography(
    names: List[str],
    groups: List[str],
    coords: List[Tuple[float, float]],
    out_png: Path,
) -> None:
    color_map = {"target": "#C73E1D", "reference": "#1f2933", "background": "#2E86AB"}
    size_map = {"target": 70, "reference": 130, "background": 45}
    fig, ax = plt.subplots(figsize=(9, 7))
    seen_groups = set()
    for (x, y), name, grp in zip(coords, names, groups):
        label = grp if grp not in seen_groups else None
        seen_groups.add(grp)
        ax.scatter(x, y, c=color_map.get(grp, "#888"),
                   s=size_map.get(grp, 50), edgecolors="white",
                   linewidths=0.6, label=label, zorder=3)
        ax.annotate(name, (x, y), fontsize=7, xytext=(4, 3),
                    textcoords="offset points", color="#333")
    ax.set_title("Antigenic cartography (sequence-based, antigenic-site distance)")
    ax.set_xlabel("MDS axis 1")
    ax.set_ylabel("MDS axis 2")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
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


# ==============================================================================
# ===  7. 계통수(NJ) + 클레이드 색칠  ==========================================
# ==============================================================================

def build_tree_png(
    aligned: Dict[str, str],
    clade_by_name: Dict[str, str],
    out_png: Path,
    out_newick: Path,
) -> None:
    # reference 좌표로 투영된 동일 길이 서열들로 MSA 구성
    msa = MultipleSeqAlignment(
        [BioSeqRecord(Seq(seq), id=name, description="") for name, seq in aligned.items()]
    )
    dm = DistanceCalculator("identity").get_distance(msa)
    tree = DistanceTreeConstructor().nj(dm)
    tree.ladderize()
    Phylo.write(tree, str(out_newick), "newick")

    # 클레이드 -> 색
    clades = sorted(set(clade_by_name.values()))
    clade_color = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(clades)}
    label_colors = {
        name: clade_color.get(clade_by_name.get(name, ""), "#333")
        for name in aligned
    }

    n_leaves = len(aligned)
    fig, ax = plt.subplots(figsize=(11, max(3, n_leaves * 0.28)))
    Phylo.draw(
        tree,
        do_show=False,
        axes=ax,
        label_colors=lambda nm: label_colors.get(nm, "#333"),
        branch_labels=None,
    )
    # 범례(클레이드 색)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=clade_color[c], markersize=8, label=c)
               for c in clades]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="clade")
    ax.set_title("Phylogenetic tree (Neighbor-Joining, identity distance)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ==============================================================================
# ===  표/리포트 저장  ==========================================================
# ==============================================================================

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
<h2>H3 넘버링 점검 (reference)</h2>
<p class="note">아래 위치의 reference 잔기가 알려진 값과 맞는지 확인하세요. 어긋나면 상단 H3_OFFSET 을 조정하세요.</p>
{table(sanity_rows, ['h3_position','antigenic_site','reference_aa'])}
<h2>클레이드 지정</h2>
{table(clade_rows, ['sample','assigned_clade','score'])}
<h2>항원부위 변이</h2>
{table(antigenic_rows, ['sample','antigenic_site','h3_position','mutation','weight'])}
<h2>약제 작용부위</h2>
{table(drug_rows, ['sample','drug','h3_position','mutation','changed','note'])}
{imgs}
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
    parser.add_argument("--outdir", default=OUTPUT_DIR)
    parser.add_argument("--h3-offset", type=int, default=None,
                        help="H3 넘버링 보정값(기본: 코드 상단 H3_OFFSET)")
    args = parser.parse_args(argv)

    global H3_OFFSET
    if args.h3_offset is not None:
        H3_OFFSET = args.h3_offset

    base = Path(__file__).resolve().parent
    outdir = (base / args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 입력 로드 -------------------------------------------------------------
    ref_path = base / args.reference
    tgt_path = base / args.target
    bg_path = base / args.background
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

    targets = [(normalize_id(n), to_protein(s)) for n, s in read_fasta(tgt_path)]
    background = []
    if bg_path.exists():
        background = [(normalize_id(n), to_protein(s)) for n, s in read_fasta(bg_path)]
    else:
        log(f"background({bg_path.name}) 없음 → 계통수는 target 만으로 그립니다.")

    # --- reference 좌표로 정렬(투영) ------------------------------------------
    aligner = make_aligner()
    ref_proj = ref_prot  # reference 자기 자신은 좌표 그대로
    proj_targets = {n: align_to_reference(ref_prot, s, aligner) for n, s in targets}
    proj_background = {n: align_to_reference(ref_prot, s, aligner) for n, s in background}
    log(f"정렬 완료: target {len(proj_targets)}개, background {len(proj_background)}개")

    # --- H3 넘버링 점검표(사용자가 OFFSET 검증용) ----------------------------
    sanity_rows = []
    for site in [145, 155, 156, 158, 159, 189, 193]:
        label, _ = ANTIGENIC_SITES.get(site, ("-", 0))
        sanity_rows.append({"h3_position": site, "antigenic_site": label,
                            "reference_aa": residue_at(ref_proj, site) or "(좌표밖)"})
    log("H3 넘버링 점검(reference 잔기): " +
        ", ".join(f"{r['h3_position']}={r['reference_aa']}" for r in sanity_rows))

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
    clade_by_name: Dict[str, str] = {}
    clade_rows = []
    for name, proj in {**proj_targets, **proj_background}.items():
        clade, score = assign_clade(proj)
        clade_by_name[name] = clade
        clade_rows.append({"sample": name, "assigned_clade": clade, "score": score})
    clade_by_name[ref_id] = assign_clade(ref_proj)[0]
    write_csv(outdir / "clade_assignments.csv", clade_rows,
              ["sample", "assigned_clade", "score"])

    # --- Antigenic cartography -------------------------------------------------
    carto_names, carto_groups, carto_proj = [], [], []
    carto_names.append(normalize_id(ref_id)); carto_groups.append("reference"); carto_proj.append(ref_proj)
    for n, p in proj_targets.items():
        carto_names.append(n); carto_groups.append("target"); carto_proj.append(p)
    for n, p in proj_background.items():
        carto_names.append(n); carto_groups.append("background"); carto_proj.append(p)
    dmat = [[antigenic_distance(a, b) for b in carto_proj] for a in carto_proj]
    coords = classical_mds(dmat)
    draw_cartography(carto_names, carto_groups, coords, outdir / "antigenic_cartography.png")
    # 좌표/거리도 CSV로
    write_csv(outdir / "antigenic_cartography_coords.csv",
              [{"sample": n, "group": g, "x": round(x, 5), "y": round(y, 5)}
               for n, g, (x, y) in zip(carto_names, carto_groups, coords)],
              ["sample", "group", "x", "y"])

    # --- 계통수 ----------------------------------------------------------------
    tree_input = {normalize_id(ref_id): ref_proj}
    tree_input.update(proj_targets)
    tree_input.update(proj_background)
    images = ["antigenic_cartography.png"]
    if len(tree_input) >= 3:
        build_tree_png(tree_input, clade_by_name,
                       outdir / "phylogenetic_tree.png",
                       outdir / "phylogenetic_tree.newick")
        images.append("phylogenetic_tree.png")
    else:
        log("서열이 3개 미만이라 계통수는 건너뜁니다(NJ 트리는 최소 3개 필요).")

    # --- 요약 리포트 -----------------------------------------------------------
    write_report(outdir / "report.html", sanity_rows, clade_rows,
                 antigenic_rows, drug_rows, images)

    log(f"완료! 결과 폴더: {outdir}")
    log(f"요약 리포트를 브라우저로 여세요: {outdir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
