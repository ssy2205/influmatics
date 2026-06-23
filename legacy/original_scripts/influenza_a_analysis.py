#!/usr/bin/env python3
"""
Influenza A retrospective sequence analysis toolkit.

This script is designed for educational and retrospective public-data analysis.
It annotates observed sequences, summarizes known markers, and visualizes
metadata patterns. It does not design new viral sequences, rank mutations for
fitness/escape, or provide experimental protocols.

Core modules:
  - optional sequence normalization: clean/filter/dedupe and reference-frame
    alignment (MAFFT when available, pure-Python Needleman-Wunsch fallback) so
    positional site lookups share one coordinate system
  - sequence QC
  - subtype/source screening against provided references or BLAST result tables
  - clade annotation from Nextclade TSV and/or user-provided marker rules
  - alignment-based phylogenetic tree rendering with a pure-Python UPGMA fallback
  - sequence-based antigenic-site distance maps against vaccine/reference strains
  - known antiviral resistance marker annotation
  - educational retrospective selection-pressure summaries
  - region/time metadata summaries for phylogeography-style visualization
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DNA_ALPHABET = set("ACGTURYKMSWBDHVN-.?")
PROTEIN_ALPHABET = set("ABCDEFGHIKLMNPQRSTVWXYZ*- .?")
GAP_CHARS = set("-.")
UNKNOWN_AA = set("XBZJUO?-.")
SUBTYPE_RE = re.compile(r"\bH\d{1,2}N\d{1,2}\b", re.IGNORECASE)
MUTATION_RE = re.compile(r"^([A-Za-z*])?(\d+)([A-Za-z*])$")


CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


@dataclass
class SeqRecord:
    seq_id: str
    sequence: str
    description: str = ""

    @property
    def norm_id(self) -> str:
        return normalize_id(self.seq_id)


@dataclass
class TreeNode:
    name: str
    height: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    size: int = 1

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def normalize_id(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"['\"\(\)\[\]]", "", value)
    value = re.sub(r"[\s/|:.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def safe_name(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "item"


def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(message: str) -> None:
    print(f"[influenza-analysis] {message}", flush=True)


def read_fasta(path: Path) -> List[SeqRecord]:
    records: List[SeqRecord] = []
    current_id: Optional[str] = None
    current_desc = ""
    chunks: List[str] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append(
                        SeqRecord(current_id, "".join(chunks).upper(), current_desc)
                    )
                current_desc = line[1:].strip()
                current_id = current_desc.split()[0] if current_desc else f"seq_{line_no}"
                chunks = []
            else:
                chunks.append(re.sub(r"\s+", "", line))

    if current_id is not None:
        records.append(SeqRecord(current_id, "".join(chunks).upper(), current_desc))

    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def read_table(path: Path, delimiter: Optional[str] = None) -> List[Dict[str, str]]:
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if delimiter is None:
            dialect = csv.Sniffer().sniff(sample)
        else:
            dialect = csv.excel_tab if delimiter == "\t" else csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"Table has no header: {path}")
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in reader]


def write_table(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fieldnames = seen or ["message"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def is_probably_nt(seq: str) -> bool:
    letters = [c for c in seq.upper() if not c.isspace()]
    if not letters:
        return False
    nt_like = sum(1 for c in letters if c in DNA_ALPHABET)
    return nt_like / len(letters) >= 0.90


def sequence_type(seq: str) -> str:
    return "nucleotide" if is_probably_nt(seq) else "protein"


def infer_subtype(text: str) -> str:
    match = SUBTYPE_RE.search(text or "")
    return match.group(0).upper() if match else ""


def pick_column(rows: Sequence[Dict[str, str]], candidates: Sequence[str]) -> Optional[str]:
    if not rows:
        return None
    lower_to_real = {key.lower(): key for key in rows[0].keys()}
    for candidate in candidates:
        if candidate.lower() in lower_to_real:
            return lower_to_real[candidate.lower()]
    return None


def metadata_by_id(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    if not rows:
        return {}
    id_col = pick_column(rows, ["seqName", "strain", "sample", "sample_id", "id", "name", "accession"])
    if id_col is None:
        id_col = list(rows[0].keys())[0]
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        norm = normalize_id(row.get(id_col, ""))
        if norm:
            out[norm] = row
    return out


def clean_sequence(seq: str) -> str:
    return re.sub(r"\s+", "", seq.upper())


def p_distance(seq_a: str, seq_b: str) -> float:
    a = clean_sequence(seq_a)
    b = clean_sequence(seq_b)
    n = min(len(a), len(b))
    compared = 0
    mismatches = 0
    for i in range(n):
        ca, cb = a[i], b[i]
        if ca in GAP_CHARS or cb in GAP_CHARS or ca == "N" or cb == "N" or ca == "X" or cb == "X":
            continue
        compared += 1
        if ca != cb:
            mismatches += 1
    if compared == 0:
        return 1.0
    length_penalty = abs(len(a) - len(b)) / max(len(a), len(b), 1)
    return min(1.0, mismatches / compared + length_penalty)


def percent_identity(seq_a: str, seq_b: str) -> float:
    return 100.0 * (1.0 - p_distance(seq_a, seq_b))


def kmer_set(seq: str, k: int = 15) -> set:
    seq = re.sub(r"[^A-Z]", "", seq.upper())
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i : i + k] for i in range(0, len(seq) - k + 1)}


def kmer_jaccard(seq_a: str, seq_b: str, k: int = 15) -> float:
    a = kmer_set(seq_a, k)
    b = kmer_set(seq_b, k)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Sequence normalization and reference-frame alignment
#
# Several downstream modules (antigenic sites, clade marker rules, antiviral
# resistance markers) index sequences positionally as seq[site - 1]. That is
# only meaningful if every sequence shares one coordinate system. This stage
# cleans/filters input, optionally collapses duplicates, aligns to a chosen
# reference (MAFFT when available, pure-Python Needleman-Wunsch otherwise) and
# projects every sequence onto the reference's ungapped coordinate frame so the
# published site numbers line up.
# ---------------------------------------------------------------------------

NT_AMBIGUOUS = set("NRYKMSWBDHV?")


@dataclass
class NormItem:
    seq_id: str
    description: str
    status: str  # reference | kept | duplicate | dropped_short | dropped_ambiguous
    clean_length: int = 0
    ambiguous_fraction: float = 0.0
    duplicate_of: str = ""
    insertions_vs_reference: int = 0
    alignment_method: str = ""


def strip_gaps(seq: str) -> str:
    return "".join(c for c in seq if c not in GAP_CHARS)


def ungapped_clean(seq: str) -> str:
    return strip_gaps(clean_sequence(seq))


def detect_record_kind(records: Sequence[SeqRecord]) -> str:
    if not records:
        return "nucleotide"
    nt_votes = sum(1 for record in records if is_probably_nt(record.sequence))
    return "nucleotide" if nt_votes * 2 >= len(records) else "protein"


def ambiguous_fraction(seq: str, kind: str) -> float:
    if not seq:
        return 1.0
    ambiguous = NT_AMBIGUOUS if kind == "nucleotide" else UNKNOWN_AA
    return sum(1 for c in seq if c in ambiguous) / len(seq)


def pick_reference_record(
    records: Sequence[SeqRecord], reference_id: Optional[str]
) -> SeqRecord:
    if not records:
        raise ValueError("No sequences available to choose an alignment reference.")
    if reference_id:
        target = normalize_id(reference_id)
        for record in records:
            if record.norm_id == target or record.seq_id == reference_id:
                return record
        raise ValueError(f"Reference id not found in input: {reference_id}")
    return max(records, key=lambda r: len(ungapped_clean(r.sequence)))


def needleman_wunsch(
    ref: str, qry: str, match: int = 1, mismatch: int = -1, gap: int = -2
) -> Tuple[str, str]:
    """Global pairwise alignment. Pure Python, 1-byte/cell traceback."""
    m, n = len(ref), len(qry)
    if m == 0 or n == 0:
        return ref + "-" * n, "-" * m + qry
    width = n + 1
    # traceback codes: 0 diagonal, 1 up (gap in qry), 2 left (gap in ref)
    tb = bytearray(width * (m + 1))
    prev = [gap * j for j in range(width)]
    for j in range(1, width):
        tb[j] = 2
    for i in range(1, m + 1):
        curr = [gap * i] + [0] * n
        row_off = i * width
        tb[row_off] = 1
        ref_c = ref[i - 1]
        for j in range(1, width):
            diag = prev[j - 1] + (match if ref_c == qry[j - 1] else mismatch)
            up = prev[j] + gap
            left = curr[j - 1] + gap
            best = diag
            code = 0
            if up > best:
                best = up
                code = 1
            if left > best:
                best = left
                code = 2
            curr[j] = best
            tb[row_off + j] = code
        prev = curr
    a_ref: List[str] = []
    a_qry: List[str] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            code = tb[i * width + j]
        else:
            code = 1 if j == 0 else 2
        if code == 0:
            a_ref.append(ref[i - 1])
            a_qry.append(qry[j - 1])
            i -= 1
            j -= 1
        elif code == 1:
            a_ref.append(ref[i - 1])
            a_qry.append("-")
            i -= 1
        else:
            a_ref.append("-")
            a_qry.append(qry[j - 1])
            j -= 1
    return "".join(reversed(a_ref)), "".join(reversed(a_qry))


def project_to_reference_frame(
    aligned_ref: str, aligned_qry: str, ref_len: int
) -> Tuple[str, int]:
    """Return (frame string of length ref_len, insertion count vs reference)."""
    frame = ["-"] * ref_len
    insertions = 0
    ref_pos = 0
    for rc, qc in zip(aligned_ref, aligned_qry):
        if rc == "-":
            if qc != "-":
                insertions += 1
            continue
        ref_pos += 1
        if ref_pos <= ref_len:
            frame[ref_pos - 1] = qc
    return "".join(frame), insertions


def run_mafft_msa(records: Sequence[SeqRecord], threads: int) -> Optional[Dict[str, str]]:
    """Run MAFFT on (already ungapped) records. Returns {seq_id: aligned} or None."""
    if not records or shutil.which("mafft") is None:
        return None
    tmpdir = tempfile.mkdtemp(prefix="influenza_align_")
    in_path = Path(tmpdir) / "input.fasta"
    out_path = Path(tmpdir) / "aligned.fasta"
    order: List[str] = []
    lines: List[str] = []
    for idx, record in enumerate(records):
        order.append(record.seq_id)
        lines.append(f">s{idx}")
        lines.append(ungapped_clean(record.sequence) or "N")
    in_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        with out_path.open("w", encoding="utf-8") as handle:
            subprocess.run(
                ["mafft", "--auto", "--thread", str(max(1, threads)), str(in_path)],
                check=True,
                stdout=handle,
                stderr=subprocess.DEVNULL,
            )
        aligned = read_fasta(out_path)
    except Exception:
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    by_tag = {record.seq_id: record.sequence for record in aligned}
    return {seq_id: by_tag.get(f"s{idx}", "") for idx, seq_id in enumerate(order)}


def frame_records_against(
    records: Sequence[SeqRecord],
    reference: SeqRecord,
    prefer_mafft: bool,
    threads: int,
) -> Tuple[List[SeqRecord], Dict[str, int], str]:
    """Project records onto the reference's ungapped coordinate frame."""
    ref_ungapped = ungapped_clean(reference.sequence)
    ref_len = len(ref_ungapped)
    insertions: Dict[str, int] = {}
    framed: List[SeqRecord] = []
    if not records:
        return framed, insertions, "needleman_wunsch_pure_python"

    msa: Optional[Dict[str, str]] = None
    used = "needleman_wunsch_pure_python"
    if prefer_mafft:
        msa = run_mafft_msa([reference] + list(records), threads)
        if msa is not None:
            used = "mafft"

    if msa is not None:
        aligned_ref = msa.get(reference.seq_id, "")
        for record in records:
            frame, ins = project_to_reference_frame(
                aligned_ref, msa.get(record.seq_id, ""), ref_len
            )
            framed.append(SeqRecord(record.seq_id, frame, record.description))
            insertions[record.seq_id] = ins
    else:
        for record in records:
            a_ref, a_qry = needleman_wunsch(ref_ungapped, ungapped_clean(record.sequence))
            frame, ins = project_to_reference_frame(a_ref, a_qry, ref_len)
            framed.append(SeqRecord(record.seq_id, frame, record.description))
            insertions[record.seq_id] = ins
    return framed, insertions, used


def normalize_and_frame(
    records: Sequence[SeqRecord],
    kind: str,
    reference_id: Optional[str],
    min_length: int,
    max_ambiguous_fraction: float,
    collapse_duplicates: bool,
    prefer_mafft: bool,
    threads: int,
) -> Tuple[List[SeqRecord], List[SeqRecord], List[Dict[str, Any]], SeqRecord, str]:
    """Clean/filter/dedupe records, then project them onto a reference frame.

    Returns (clean_records, frame_records, report_rows, reference, method).
    ``clean_records`` are ungapped (raw quality, for QC/screening). Each
    ``frame_records`` entry has length == reference ungapped length so that
    positional ``seq[site - 1]`` lookups use reference coordinates.
    """
    if kind == "auto":
        kind = detect_record_kind(records)
    reference = pick_reference_record(records, reference_id)
    ref_clean = ungapped_clean(reference.sequence)

    items: List[NormItem] = []
    survivors: List[SeqRecord] = []
    seen: Dict[str, str] = {}
    if collapse_duplicates:
        # Seed with the reference so exact duplicates of it are collapsed too.
        seen[ref_clean] = reference.seq_id

    for record in records:
        clean = ungapped_clean(record.sequence)
        amb = ambiguous_fraction(clean, kind)
        item = NormItem(
            seq_id=record.seq_id,
            description=record.description,
            status="kept",
            clean_length=len(clean),
            ambiguous_fraction=round(amb, 6),
        )
        if record is reference:
            item.status = "reference"
            items.append(item)
            continue
        if min_length and len(clean) < min_length:
            item.status = "dropped_short"
            items.append(item)
            continue
        if max_ambiguous_fraction < 1.0 and amb > max_ambiguous_fraction:
            item.status = "dropped_ambiguous"
            items.append(item)
            continue
        if collapse_duplicates and clean in seen:
            item.status = "duplicate"
            item.duplicate_of = seen[clean]
            items.append(item)
            continue
        if collapse_duplicates:
            seen[clean] = record.seq_id
        survivors.append(record)
        items.append(item)

    framed_survivors, insertions, used = frame_records_against(
        survivors, reference, prefer_mafft, threads
    )

    clean_records: List[SeqRecord] = [
        SeqRecord(reference.seq_id, ref_clean, reference.description)
    ]
    frame_records: List[SeqRecord] = [
        SeqRecord(reference.seq_id, ref_clean, reference.description)
    ]
    framed_by_id = {record.seq_id: record for record in framed_survivors}
    for record in survivors:
        clean_records.append(
            SeqRecord(record.seq_id, ungapped_clean(record.sequence), record.description)
        )
        frame_records.append(framed_by_id[record.seq_id])

    for item in items:
        if item.status == "kept":
            item.insertions_vs_reference = insertions.get(item.seq_id, 0)
            item.alignment_method = used
        elif item.status == "reference":
            item.alignment_method = used

    report = [
        {
            "seq_id": item.seq_id,
            "status": item.status,
            "kind": kind,
            "clean_length": item.clean_length,
            "ambiguous_fraction": item.ambiguous_fraction,
            "duplicate_of": item.duplicate_of,
            "insertions_vs_reference": item.insertions_vs_reference,
            "alignment_method": item.alignment_method,
            "reference_id": reference.seq_id,
            "reference_length": len(ref_clean),
            "interpretation": "reference_frame_normalization_for_positional_modules",
        }
        for item in items
    ]
    return clean_records, frame_records, report, reference, used


def write_fasta(path: Path, records: Sequence[SeqRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f">{record.seq_id}\n{record.sequence}\n" for record in records),
        encoding="utf-8",
    )


def compute_qc(records: Sequence[SeqRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        seq = clean_sequence(record.sequence)
        seq_kind = sequence_type(seq)
        if seq_kind == "nucleotide":
            valid = DNA_ALPHABET
            ambiguous = set("NRYKMSWBDHV?")
        else:
            valid = PROTEIN_ALPHABET
            ambiguous = UNKNOWN_AA
        invalid_count = sum(1 for c in seq if c not in valid)
        ambiguous_count = sum(1 for c in seq if c in ambiguous)
        gap_count = sum(1 for c in seq if c in GAP_CHARS)
        rows.append(
            {
                "seq_id": record.seq_id,
                "normalized_id": record.norm_id,
                "sequence_type": seq_kind,
                "length": len(seq),
                "gap_count": gap_count,
                "ambiguous_count": ambiguous_count,
                "invalid_count": invalid_count,
                "ambiguous_fraction": round(ambiguous_count / max(len(seq), 1), 6),
                "inferred_subtype_from_header": infer_subtype(record.description),
            }
        )
    return rows


def summarize_qc(qc_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [int(row["length"]) for row in qc_rows]
    return {
        "sequence_count": len(qc_rows),
        "length_min": min(lengths) if lengths else 0,
        "length_median": statistics.median(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "sequence_types": dict(Counter(row["sequence_type"] for row in qc_rows)),
        "records_with_invalid_characters": sum(1 for row in qc_rows if int(row["invalid_count"]) > 0),
    }


def screen_reference_similarity(
    query_records: Sequence[SeqRecord],
    reference_records: Sequence[SeqRecord],
    metadata: Dict[str, Dict[str, str]],
    expected_subtype_col: Optional[str],
    kmer_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not reference_records:
        return rows

    reference_info = []
    for ref in reference_records:
        subtype = infer_subtype(ref.description) or infer_subtype(ref.seq_id)
        reference_info.append((ref, subtype))

    for query in query_records:
        best: Optional[Tuple[SeqRecord, str, float, float]] = None
        for ref, ref_subtype in reference_info:
            identity = percent_identity(query.sequence, ref.sequence)
            jaccard = kmer_jaccard(query.sequence, ref.sequence, k=kmer_size)
            score = max(identity / 100.0, jaccard)
            if best is None or score > best[3]:
                best = (ref, ref_subtype, identity, score)

        meta = metadata.get(query.norm_id, {})
        expected = ""
        if expected_subtype_col and expected_subtype_col in meta:
            expected = meta.get(expected_subtype_col, "")
        if not expected:
            expected = infer_subtype(query.description)

        best_ref, best_subtype, best_identity, best_score = best  # type: ignore[misc]
        subtype_mismatch = bool(expected and best_subtype and expected.upper() != best_subtype.upper())
        rows.append(
            {
                "seq_id": query.seq_id,
                "expected_subtype": expected,
                "best_reference": best_ref.seq_id,
                "best_reference_subtype": best_subtype,
                "best_identity_percent": round(best_identity, 3),
                "similarity_score": round(best_score, 5),
                "subtype_mismatch_flag": subtype_mismatch,
                "interpretation": "review_subtype_or_source_metadata" if subtype_mismatch else "no_subtype_mismatch_detected",
            }
        )
    return rows


def summarize_blast_results(path: Path) -> List[Dict[str, Any]]:
    rows = read_table(path)
    out: List[Dict[str, Any]] = []
    for row in rows:
        query = row.get("query_name") or row.get("query") or row.get("seq_id") or ""
        title = row.get("subject_title") or row.get("subject") or row.get("title") or ""
        identity = row.get("identity_percentage") or row.get("pident") or ""
        out.append(
            {
                "seq_id": query,
                "blast_top_hit_title": title,
                "blast_top_hit_subtype": infer_subtype(title),
                "identity_percentage": identity,
                "note": "BLAST result parsed from user-provided table; no online BLAST was run.",
            }
        )
    return out


def load_nextclade_clades(path: Path) -> List[Dict[str, Any]]:
    rows = read_table(path)
    seq_col = pick_column(rows, ["seqName", "seq_name", "name", "strain", "id"])
    clade_col = pick_column(rows, ["clade", "legacy.clade", "Nextclade_pango", "nextclade.clade"])
    qc_col = pick_column(rows, ["qc.overallStatus", "qc_status", "qc.overallScore"])
    if seq_col is None:
        raise ValueError("Nextclade TSV/CSV needs a seqName/name/id column.")
    out = []
    for row in rows:
        out.append(
            {
                "seq_id": row.get(seq_col, ""),
                "normalized_id": normalize_id(row.get(seq_col, "")),
                "nextclade_clade": row.get(clade_col, "") if clade_col else "",
                "nextclade_qc": row.get(qc_col, "") if qc_col else "",
            }
        )
    return out


def parse_allowed_residues(value: str) -> List[str]:
    value = str(value or "").strip().upper()
    if not value:
        return []
    value = value.replace("|", "/").replace(";", "/").replace(",", "/")
    parts = [part.strip() for part in value.split("/") if part.strip()]
    if len(parts) == 1 and len(parts[0]) > 1 and all(c.isalpha() or c == "*" for c in parts[0]):
        return list(parts[0])
    return parts


def classify_by_clade_rules(
    protein_records: Sequence[SeqRecord],
    rule_rows: Sequence[Dict[str, str]],
    min_score: float = 0.75,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rule_rows:
        clade = row.get("clade") or row.get("Clade") or row.get("label") or ""
        if clade:
            grouped[clade].append(row)

    rows: List[Dict[str, Any]] = []
    for record in protein_records:
        seq = clean_sequence(record.sequence)
        scores = []
        for clade, rules in grouped.items():
            tested = 0
            matched = 0
            details = []
            for rule in rules:
                site_raw = rule.get("site") or rule.get("position") or rule.get("aa_site") or ""
                aa_raw = rule.get("aa") or rule.get("residue") or rule.get("mutation") or ""
                try:
                    site = int(float(site_raw))
                except ValueError:
                    continue
                if site < 1 or site > len(seq):
                    continue
                allowed = parse_allowed_residues(aa_raw)
                observed = seq[site - 1]
                if not allowed:
                    continue
                tested += 1
                is_match = observed in allowed
                matched += 1 if is_match else 0
                details.append(f"{site}{observed}:{'match' if is_match else 'no'}")
            score = matched / tested if tested else 0.0
            scores.append((score, matched, tested, clade, details))
        scores.sort(reverse=True)
        best = scores[0] if scores else (0.0, 0, 0, "", [])
        rows.append(
            {
                "seq_id": record.seq_id,
                "normalized_id": record.norm_id,
                "rule_based_clade": best[3] if best[0] >= min_score else "",
                "rule_based_clade_score": round(best[0], 4),
                "matched_markers": best[1],
                "tested_markers": best[2],
                "marker_details": ";".join(best[4][:50]),
                "interpretation": "rule_threshold_met" if best[0] >= min_score else "no_rule_threshold_met",
            }
        )
    return rows


def merge_clade_results(
    records: Sequence[SeqRecord],
    nextclade_rows: Sequence[Dict[str, Any]],
    rule_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    next_by_id = {row["normalized_id"]: row for row in nextclade_rows}
    rule_by_id = {row["normalized_id"]: row for row in rule_rows}
    out: List[Dict[str, Any]] = []
    for record in records:
        next_row = next_by_id.get(record.norm_id, {})
        rule_row = rule_by_id.get(record.norm_id, {})
        final = next_row.get("nextclade_clade") or rule_row.get("rule_based_clade") or ""
        source = "nextclade" if next_row.get("nextclade_clade") else ("marker_rules" if rule_row.get("rule_based_clade") else "")
        out.append(
            {
                "seq_id": record.seq_id,
                "normalized_id": record.norm_id,
                "final_clade": final,
                "clade_source": source,
                "nextclade_clade": next_row.get("nextclade_clade", ""),
                "nextclade_qc": next_row.get("nextclade_qc", ""),
                "rule_based_clade": rule_row.get("rule_based_clade", ""),
                "rule_based_clade_score": rule_row.get("rule_based_clade_score", ""),
            }
        )
    return out


def load_antigenic_sites(path: Path) -> List[Dict[str, Any]]:
    rows = read_table(path)
    out = []
    for row in rows:
        site_raw = row.get("site") or row.get("position") or row.get("aa_site") or ""
        try:
            site = int(float(site_raw))
        except ValueError:
            continue
        label = row.get("label") or row.get("site_label") or row.get("antigenic_site") or f"site_{site}"
        try:
            weight = float(row.get("weight", "1") or "1")
        except ValueError:
            weight = 1.0
        out.append(
            {
                "site": site,
                "label": label,
                "weight": weight,
                "source": row.get("source", ""),
                "note": row.get("note", ""),
            }
        )
    if not out:
        raise ValueError("No valid antigenic sites found. Required column: site or position.")
    return out


def antigenic_site_distance(
    seq_a: str,
    seq_b: str,
    sites: Sequence[Dict[str, Any]],
) -> Tuple[float, int, int, float]:
    total_weight = 0.0
    mismatch_weight = 0.0
    compared = 0
    mismatches = 0
    a = clean_sequence(seq_a)
    b = clean_sequence(seq_b)
    for site in sites:
        pos = int(site["site"])
        if pos < 1 or pos > len(a) or pos > len(b):
            continue
        aa = a[pos - 1]
        bb = b[pos - 1]
        if aa in UNKNOWN_AA or bb in UNKNOWN_AA:
            continue
        weight = float(site.get("weight", 1.0))
        total_weight += weight
        compared += 1
        if aa != bb:
            mismatch_weight += weight
            mismatches += 1
    distance = mismatch_weight / total_weight if total_weight else 0.0
    return distance, compared, mismatches, total_weight


def compute_antigenic_distances(
    protein_records: Sequence[SeqRecord],
    vaccine_records: Sequence[SeqRecord],
    antigenic_sites: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pair_rows: List[Dict[str, Any]] = []
    site_rows: List[Dict[str, Any]] = []
    for sample in protein_records:
        for vaccine in vaccine_records:
            distance, compared, mismatches, total_weight = antigenic_site_distance(
                sample.sequence, vaccine.sequence, antigenic_sites
            )
            pair_rows.append(
                {
                    "seq_id": sample.seq_id,
                    "vaccine_id": vaccine.seq_id,
                    "antigenic_site_distance": round(distance, 6),
                    "sites_compared": compared,
                    "site_mismatches": mismatches,
                    "total_site_weight": round(total_weight, 3),
                    "interpretation": "sequence_based_antigenic_site_distance_not_HI_cartography",
                }
            )
            for site in antigenic_sites:
                pos = int(site["site"])
                sample_aa = sample.sequence[pos - 1] if 1 <= pos <= len(sample.sequence) else ""
                vaccine_aa = vaccine.sequence[pos - 1] if 1 <= pos <= len(vaccine.sequence) else ""
                if sample_aa and vaccine_aa and sample_aa not in UNKNOWN_AA and vaccine_aa not in UNKNOWN_AA:
                    site_rows.append(
                        {
                            "seq_id": sample.seq_id,
                            "vaccine_id": vaccine.seq_id,
                            "site": pos,
                            "site_label": site.get("label", ""),
                            "sample_residue": sample_aa,
                            "vaccine_residue": vaccine_aa,
                            "mismatch": sample_aa != vaccine_aa,
                            "weight": site.get("weight", 1.0),
                            "source": site.get("source", ""),
                        }
                    )
    return pair_rows, site_rows


def classical_mds(distance_matrix: List[List[float]]) -> Optional[List[Tuple[float, float]]]:
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    if not distance_matrix:
        return []
    d = np.array(distance_matrix, dtype=float)
    n = d.shape[0]
    if n == 1:
        return [(0.0, 0.0)]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j.dot(d ** 2).dot(j)
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    coords = np.zeros((n, 2))
    for axis in range(min(2, n)):
        if vals[axis] > 0:
            coords[:, axis] = vecs[:, axis] * math.sqrt(float(vals[axis]))
    return [(round(float(x), 6), round(float(y), 6)) for x, y in coords]


def anchor_projection(distance_matrix: List[List[float]]) -> List[Tuple[float, float]]:
    n = len(distance_matrix)
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]
    farthest = (0, 1, distance_matrix[0][1])
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i][j] > farthest[2]:
                farthest = (i, j, distance_matrix[i][j])
    a, b, dab = farthest
    dab = max(dab, 1e-9)
    coords = []
    for idx in range(n):
        da = distance_matrix[idx][a]
        db = distance_matrix[idx][b]
        x = (da * da + dab * dab - db * db) / (2 * dab)
        y_sq = max(0.0, da * da - x * x)
        sign = -1.0 if hash(idx) % 2 else 1.0
        coords.append((round(x, 6), round(sign * math.sqrt(y_sq), 6)))
    return coords


def compute_antigenic_map(
    protein_records: Sequence[SeqRecord],
    vaccine_records: Sequence[SeqRecord],
    antigenic_sites: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    all_records = list(vaccine_records) + list(protein_records)
    labels = [record.seq_id for record in all_records]
    groups = ["vaccine_or_reference"] * len(vaccine_records) + ["sample"] * len(protein_records)
    matrix: List[List[float]] = []
    for rec_a in all_records:
        row = []
        for rec_b in all_records:
            distance, _, _, _ = antigenic_site_distance(rec_a.sequence, rec_b.sequence, antigenic_sites)
            row.append(distance)
        matrix.append(row)
    coords = classical_mds(matrix)
    method = "classical_mds_numpy"
    if coords is None:
        coords = anchor_projection(matrix)
        method = "two_anchor_distance_projection"
    out = []
    for label, group, (x, y) in zip(labels, groups, coords):
        out.append(
            {
                "seq_id": label,
                "group": group,
                "x": x,
                "y": y,
                "projection_method": method,
                "interpretation": "sequence_based_projection_not_serologic_antigenic_cartography",
            }
        )
    return out


def parse_marker_row(row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    mutation_field = (
        row.get("mutation")
        or row.get("marker")
        or row.get("aa_change")
        or row.get("substitution")
        or ""
    ).strip()
    site_raw = row.get("site") or row.get("position") or row.get("aa_site") or ""
    wildtype = (row.get("wildtype") or row.get("ref") or "").strip().upper()
    mutations: List[str] = []
    site: Optional[int] = None

    if site_raw:
        try:
            site = int(float(site_raw))
        except ValueError:
            site = None
    match = MUTATION_RE.match(mutation_field.upper())
    if match:
        if match.group(1):
            wildtype = match.group(1).upper()
        site = int(match.group(2))
        mutations = [match.group(3).upper()]
    elif mutation_field:
        mutations = parse_allowed_residues(mutation_field)

    if site is None or not mutations:
        return None
    return {
        "gene": (row.get("gene") or row.get("segment") or "").strip().upper(),
        "site": site,
        "wildtype": wildtype,
        "mutations": mutations,
        "drug": row.get("drug") or row.get("antiviral") or "",
        "effect": row.get("effect") or row.get("phenotype") or "",
        "evidence": row.get("evidence") or "",
        "source": row.get("source") or row.get("reference") or "",
        "note": row.get("note") or "",
    }


def annotate_resistance_markers(
    protein_records: Sequence[SeqRecord],
    marker_rows: Sequence[Dict[str, str]],
    protein_gene: str,
) -> List[Dict[str, Any]]:
    markers = [marker for marker in (parse_marker_row(row) for row in marker_rows) if marker]
    gene = protein_gene.upper()
    out: List[Dict[str, Any]] = []
    for record in protein_records:
        seq = clean_sequence(record.sequence)
        for marker in markers:
            marker_gene = marker["gene"]
            if marker_gene and gene and marker_gene != gene:
                continue
            site = int(marker["site"])
            if site < 1 or site > len(seq):
                continue
            observed = seq[site - 1]
            present = observed in marker["mutations"]
            if present:
                out.append(
                    {
                        "seq_id": record.seq_id,
                        "gene": marker_gene or gene,
                        "site": site,
                        "observed_residue": observed,
                        "known_marker_residue": "/".join(marker["mutations"]),
                        "wildtype_residue": marker["wildtype"],
                        "drug": marker["drug"],
                        "effect": marker["effect"],
                        "evidence": marker["evidence"],
                        "source": marker["source"],
                        "note": marker["note"],
                        "interpretation": "known_published_marker_annotation_only",
                    }
                )
    return out


def valid_codon(codon: str) -> bool:
    codon = codon.upper().replace("U", "T")
    return len(codon) == 3 and codon in CODON_TABLE and CODON_TABLE[codon] != ""


def translate_codon(codon: str) -> str:
    return CODON_TABLE.get(codon.upper().replace("U", "T"), "X")


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def analyze_selection_pressure(cds_records: Sequence[SeqRecord]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not cds_records:
        return [], {}
    lengths = {len(clean_sequence(record.sequence)) for record in cds_records}
    if len(lengths) != 1:
        raise ValueError("Selection analysis requires an aligned CDS FASTA with equal sequence lengths.")
    seq_len = next(iter(lengths))
    codon_count = seq_len // 3
    site_rows: List[Dict[str, Any]] = []
    total_syn = 0
    total_nonsyn = 0
    total_pairs = 0

    for codon_index in range(codon_count):
        codons = []
        start = codon_index * 3
        for record in cds_records:
            codon = clean_sequence(record.sequence[start : start + 3]).replace("U", "T")
            if valid_codon(codon):
                codons.append(codon)
        counts = Counter(codons)
        syn_changes = 0
        nonsyn_changes = 0
        multi_nt_pairs = 0
        compared_pairs = 0
        unique_codons = list(counts.keys())
        for i, codon_a in enumerate(unique_codons):
            for codon_b in unique_codons[i + 1 :]:
                pair_weight = counts[codon_a] * counts[codon_b]
                compared_pairs += pair_weight
                if hamming(codon_a, codon_b) > 1:
                    multi_nt_pairs += pair_weight
                if translate_codon(codon_a) == translate_codon(codon_b):
                    syn_changes += pair_weight
                else:
                    nonsyn_changes += pair_weight
        ratio = (nonsyn_changes + 0.5) / (syn_changes + 0.5)
        total_syn += syn_changes
        total_nonsyn += nonsyn_changes
        total_pairs += compared_pairs
        site_rows.append(
            {
                "codon_site": codon_index + 1,
                "valid_codon_observations": len(codons),
                "unique_codons": len(unique_codons),
                "compared_pairs": compared_pairs,
                "synonymous_pair_changes": syn_changes,
                "nonsynonymous_pair_changes": nonsyn_changes,
                "multi_nt_difference_pairs": multi_nt_pairs,
                "educational_pairwise_nonsyn_syn_ratio": round(ratio, 6),
                "interpretation": "retrospective_summary_not_a_fitness_or_escape_score",
            }
        )

    summary = {
        "sequence_count": len(cds_records),
        "codon_sites": codon_count,
        "compared_pairs_total": total_pairs,
        "synonymous_pair_changes_total": total_syn,
        "nonsynonymous_pair_changes_total": total_nonsyn,
        "educational_pairwise_nonsyn_syn_ratio_total": round((total_nonsyn + 0.5) / (total_syn + 0.5), 6),
        "method_note": "Simple pairwise codon-change summary. Use HyPhy/Datamonkey for publication-grade dN/dS inference.",
    }
    return site_rows, summary


def parse_date_to_bin(value: str, bin_size: str) -> str:
    value = str(value or "").strip()
    if not value:
        return "unknown"
    year_match = re.match(r"^(\d{4})", value)
    if not year_match:
        return "unknown"
    year = int(year_match.group(1))
    if bin_size == "year":
        return str(year)
    month = 1
    month_match = re.match(r"^\d{4}[-/](\d{1,2})", value)
    if month_match:
        month = max(1, min(12, int(month_match.group(1))))
    if bin_size == "quarter":
        quarter = (month - 1) // 3 + 1
        return f"{year}-Q{quarter}"
    return f"{year}-{month:02d}"


def summarize_phylogeography(
    metadata_rows: Sequence[Dict[str, str]],
    clade_rows: Sequence[Dict[str, Any]],
    date_column: Optional[str],
    region_column: Optional[str],
    bin_size: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not metadata_rows:
        return [], []
    id_col = pick_column(metadata_rows, ["seqName", "strain", "sample", "sample_id", "id", "name", "accession"])
    if id_col is None:
        id_col = list(metadata_rows[0].keys())[0]
    date_col = date_column or pick_column(metadata_rows, ["date", "collection_date", "sample_date", "year"])
    region_col = region_column or pick_column(metadata_rows, ["region", "country", "division", "location", "state"])
    clade_by_id = {row["normalized_id"]: row.get("final_clade", "") for row in clade_rows}

    counts: Counter = Counter()
    region_counts: Counter = Counter()
    for row in metadata_rows:
        norm = normalize_id(row.get(id_col, ""))
        time_bin = parse_date_to_bin(row.get(date_col, "") if date_col else "", bin_size)
        region = row.get(region_col, "") if region_col else ""
        region = region or "unknown"
        clade = clade_by_id.get(norm, "") or "unassigned"
        counts[(time_bin, region, clade)] += 1
        region_counts[(region, clade)] += 1

    count_rows = [
        {"time_bin": key[0], "region": key[1], "clade": key[2], "count": value}
        for key, value in sorted(counts.items())
    ]
    region_rows = [
        {"region": key[0], "clade": key[1], "count": value}
        for key, value in sorted(region_counts.items())
    ]
    return count_rows, region_rows


def distance_matrix(records: Sequence[SeqRecord]) -> List[List[float]]:
    matrix = []
    for rec_a in records:
        row = []
        for rec_b in records:
            row.append(p_distance(rec_a.sequence, rec_b.sequence))
        matrix.append(row)
    return matrix


def build_upgma_tree(records: Sequence[SeqRecord]) -> Optional[TreeNode]:
    if not records:
        return None
    if len(records) == 1:
        return TreeNode(records[0].seq_id)

    nodes: Dict[str, TreeNode] = {
        str(i): TreeNode(name=record.seq_id, height=0.0, size=1)
        for i, record in enumerate(records)
    }
    dist: Dict[Tuple[str, str], float] = {}
    matrix = distance_matrix(records)
    keys = list(nodes.keys())
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i < j:
                dist[tuple(sorted((ki, kj)))] = matrix[i][j]

    next_id = len(nodes)
    while len(nodes) > 1:
        pair, min_dist = min(dist.items(), key=lambda item: item[1])
        a, b = pair
        node_a = nodes[a]
        node_b = nodes[b]
        new_key = str(next_id)
        next_id += 1
        new_node = TreeNode(
            name=f"node_{new_key}",
            height=min_dist / 2.0,
            left=node_a,
            right=node_b,
            size=node_a.size + node_b.size,
        )

        remaining = [key for key in nodes.keys() if key not in {a, b}]
        new_distances: Dict[Tuple[str, str], float] = {}
        for key in remaining:
            da = dist.get(tuple(sorted((a, key))), 0.0)
            db = dist.get(tuple(sorted((b, key))), 0.0)
            weighted = (da * node_a.size + db * node_b.size) / (node_a.size + node_b.size)
            new_distances[tuple(sorted((new_key, key)))] = weighted

        dist = {
            key: value
            for key, value in dist.items()
            if a not in key and b not in key
        }
        dist.update(new_distances)
        del nodes[a]
        del nodes[b]
        nodes[new_key] = new_node

    return next(iter(nodes.values()))


def node_to_newick(node: TreeNode, parent_height: Optional[float] = None) -> str:
    branch = ""
    if parent_height is not None:
        branch_len = max(parent_height - node.height, 0.0)
        branch = f":{branch_len:.6f}"
    if node.is_leaf():
        return f"{quote_newick_name(node.name)}{branch}"
    left = node_to_newick(node.left, node.height) if node.left else ""
    right = node_to_newick(node.right, node.height) if node.right else ""
    return f"({left},{right}){quote_newick_name(node.name)}{branch}"


def quote_newick_name(name: str) -> str:
    if re.search(r"[\s,:;()\[\]']", name):
        return "'" + name.replace("'", "_") + "'"
    return name


def leaf_order(node: TreeNode) -> List[TreeNode]:
    if node.is_leaf():
        return [node]
    out: List[TreeNode] = []
    if node.left:
        out.extend(leaf_order(node.left))
    if node.right:
        out.extend(leaf_order(node.right))
    return out


def tree_edges(node: TreeNode) -> List[Tuple[TreeNode, TreeNode]]:
    edges = []
    for child in [node.left, node.right]:
        if child:
            edges.append((node, child))
            edges.extend(tree_edges(child))
    return edges


def render_tree_svg(
    node: TreeNode,
    output_path: Path,
    clade_by_id: Optional[Dict[str, str]] = None,
    width: int = 1400,
) -> None:
    leaves = leaf_order(node)
    y_step = 18
    margin_left = 35
    margin_right = 360
    margin_top = 30
    height = max(160, margin_top * 2 + y_step * max(1, len(leaves)))
    plot_width = width - margin_left - margin_right
    root_height = max(node.height, 1e-9)

    y_pos: Dict[int, float] = {}
    for idx, leaf in enumerate(leaves):
        y_pos[id(leaf)] = margin_top + idx * y_step

    def assign_internal_y(current: TreeNode) -> float:
        if current.is_leaf():
            return y_pos[id(current)]
        child_ys = []
        if current.left:
            child_ys.append(assign_internal_y(current.left))
        if current.right:
            child_ys.append(assign_internal_y(current.right))
        y = sum(child_ys) / len(child_ys)
        y_pos[id(current)] = y
        return y

    assign_internal_y(node)

    def x_pos(current: TreeNode) -> float:
        return margin_left + (root_height - current.height) / root_height * plot_width

    palette = [
        "#2E86AB", "#F18F01", "#C73E1D", "#3B8EA5", "#6A4C93",
        "#4CAF50", "#B56576", "#5C677D", "#D95D39", "#1B998B",
    ]
    clades = sorted(set((clade_by_id or {}).values()) - {""})
    clade_colors = {clade: palette[i % len(palette)] for i, clade in enumerate(clades)}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:11px}.muted{fill:#667}</style>',
        '<text x="24" y="18" class="muted">UPGMA tree from aligned sequences; for publication-grade inference, compare with IQ-TREE/FastTree/Nextstrain.</text>',
    ]
    for parent, child in tree_edges(node):
        px, py = x_pos(parent), y_pos[id(parent)]
        cx, cy = x_pos(child), y_pos[id(child)]
        color = "#777"
        if child.is_leaf() and clade_by_id:
            color = clade_colors.get(clade_by_id.get(normalize_id(child.name), ""), "#777")
        parts.append(f'<line x1="{px:.2f}" y1="{py:.2f}" x2="{px:.2f}" y2="{cy:.2f}" stroke="{color}" stroke-width="1"/>')
        parts.append(f'<line x1="{px:.2f}" y1="{cy:.2f}" x2="{cx:.2f}" y2="{cy:.2f}" stroke="{color}" stroke-width="1"/>')
    for leaf in leaves:
        x, y = x_pos(leaf), y_pos[id(leaf)]
        clade = (clade_by_id or {}).get(normalize_id(leaf.name), "")
        color = clade_colors.get(clade, "#333")
        label = html.escape(leaf.name)
        suffix = f" ({html.escape(clade)})" if clade else ""
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{color}"/>')
        parts.append(f'<text x="{x + 6:.2f}" y="{y + 3:.2f}" fill="{color}">{label}{suffix}</text>')
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_antigenic_map_html(path: Path, coord_rows: Sequence[Dict[str, Any]]) -> None:
    data = json.dumps(list(coord_rows), ensure_ascii=False)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sequence-based antigenic-site map</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    .note {{ color: #52616b; max-width: 920px; line-height: 1.45; }}
    #chart {{ width: min(1100px, 96vw); height: 720px; }}
  </style>
</head>
<body>
  <h1>Sequence-based antigenic-site projection</h1>
  <p class="note">This is a sequence-distance visualization over published antigenic sites.
  It is not serologic HI-based antigenic cartography and should be interpreted as a retrospective descriptive map.</p>
  <div id="chart"></div>
  <script>
    const rows = {data};
    const groups = [...new Set(rows.map(r => r.group))];
    const traces = groups.map(g => {{
      const subset = rows.filter(r => r.group === g);
      return {{
        type: "scatter",
        mode: "markers+text",
        name: g,
        x: subset.map(r => r.x),
        y: subset.map(r => r.y),
        text: subset.map(r => r.seq_id),
        textposition: "top center",
        marker: {{ size: g === "vaccine_or_reference" ? 13 : 8, line: {{ width: 1, color: "#1f2933" }} }}
      }};
    }});
    Plotly.newPlot("chart", traces, {{
      template: "plotly_white",
      xaxis: {{ title: "projection axis 1" }},
      yaxis: {{ title: "projection axis 2" }},
      legend: {{ orientation: "h" }},
      margin: {{ l: 60, r: 20, t: 20, b: 60 }}
    }}, {{ responsive: true }});
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_dashboard_html(
    path: Path,
    title: str,
    manifest: Dict[str, Any],
    chart_data: Dict[str, Any],
    tree_svg_path: Optional[Path] = None,
) -> None:
    data = json.dumps(chart_data, ensure_ascii=False)
    tree_svg = ""
    if tree_svg_path and tree_svg_path.exists():
        tree_svg = tree_svg_path.read_text(encoding="utf-8")
    manifest_html = html.escape(json.dumps(manifest, indent=2, ensure_ascii=False))
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #52616b;
      --line: #d9e2ec;
      --band: #f5f7fa;
      --accent: #2E86AB;
    }}
    body {{ margin: 0; font-family: Arial, sans-serif; color: var(--ink); background: white; }}
    header {{ padding: 28px 36px 18px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 30px 0 12px; font-size: 19px; letter-spacing: 0; }}
    main {{ padding: 0 36px 36px; max-width: 1500px; }}
    .note {{ color: var(--muted); max-width: 1000px; line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}
    .panel {{ border: 1px solid var(--line); border-radius: 8px; padding: 16px; background: #fff; }}
    .chart {{ height: 420px; }}
    pre {{ background: var(--band); border: 1px solid var(--line); padding: 14px; overflow: auto; border-radius: 8px; }}
    .tree-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="note">Retrospective educational analysis of observed Influenza A sequences. Results annotate known patterns and should not be interpreted as experimental guidance, clinical advice, or mutation design.</p>
  </header>
  <main>
    <div class="grid">
      <section class="panel">
        <h2>Sequence QC</h2>
        <div id="qc_lengths" class="chart"></div>
      </section>
      <section class="panel">
        <h2>Clade Counts</h2>
        <div id="clade_counts" class="chart"></div>
      </section>
      <section class="panel">
        <h2>Region-Time Counts</h2>
        <div id="geo_counts" class="chart"></div>
      </section>
      <section class="panel">
        <h2>Selection Summary</h2>
        <div id="selection_sites" class="chart"></div>
      </section>
    </div>
    <h2>Phylogenetic Tree</h2>
    <div class="tree-wrap">{tree_svg or '<p class="note" style="padding:16px">Tree SVG was not generated.</p>'}</div>
    <h2>Run Manifest</h2>
    <pre>{manifest_html}</pre>
  </main>
  <script>
    const data = {data};
    function countBy(rows, key) {{
      const m = new Map();
      rows.forEach(r => {{
        const k = r[key] || "unassigned";
        m.set(k, (m.get(k) || 0) + 1);
      }});
      return Array.from(m.entries()).map(([label, count]) => ({{ label, count }}));
    }}
    function plotBar(id, labels, values, title) {{
      Plotly.newPlot(id, [{{ type: "bar", x: labels, y: values, marker: {{ color: "#2E86AB" }} }}], {{
        template: "plotly_white",
        title: {{ text: title, font: {{ size: 13 }} }},
        margin: {{ l: 55, r: 15, t: 40, b: 85 }},
        xaxis: {{ tickangle: -35 }},
        yaxis: {{ title: "count" }}
      }}, {{ responsive: true }});
    }}
    const qc = data.qc || [];
    Plotly.newPlot("qc_lengths", [{{ type: "histogram", x: qc.map(r => Number(r.length)), marker: {{ color: "#1B998B" }} }}], {{
      template: "plotly_white", margin: {{ l: 55, r: 15, t: 35, b: 55 }},
      xaxis: {{ title: "sequence length" }}, yaxis: {{ title: "count" }}
    }}, {{ responsive: true }});
    const clades = countBy(data.clades || [], "final_clade");
    plotBar("clade_counts", clades.map(r => r.label), clades.map(r => r.count), "Assigned clades");
    const geo = data.phylogeography_counts || [];
    const geoLabels = geo.map(r => `${{r.time_bin}} | ${{r.region}} | ${{r.clade}}`);
    plotBar("geo_counts", geoLabels, geo.map(r => Number(r.count)), "Sampled sequences by time, region, clade");
    const sel = data.selection_sites || [];
    Plotly.newPlot("selection_sites", [{{ type: "scatter", mode: "lines+markers",
      x: sel.map(r => Number(r.codon_site)),
      y: sel.map(r => Number(r.educational_pairwise_nonsyn_syn_ratio)),
      marker: {{ color: "#C73E1D", size: 5 }}
    }}], {{
      template: "plotly_white", margin: {{ l: 55, r: 15, t: 35, b: 55 }},
      xaxis: {{ title: "codon site" }},
      yaxis: {{ title: "educational nonsyn/syn pair ratio" }}
    }}, {{ responsive: true }});
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_templates(outdir: Path) -> None:
    ensure_outdir(outdir)
    write_table(
        outdir / "metadata_template.csv",
        [
            {
                "seqName": "A/Sample/001/2024",
                "date": "2024-01-15",
                "country": "Republic of Korea",
                "subtype": "H3N2",
                "segment": "HA",
            }
        ],
    )
    write_table(
        outdir / "antigenic_sites_template.csv",
        [
            {
                "site": 145,
                "label": "published_site_example",
                "weight": 1,
                "source": "Replace with textbook/paper citation",
                "note": "Use subtype- and numbering-specific published sites.",
            }
        ],
    )
    write_table(
        outdir / "resistance_markers_template.csv",
        [
            {
                "gene": "NA",
                "mutation": "H275Y",
                "drug": "oseltamivir",
                "effect": "reported reduced susceptibility marker",
                "evidence": "published/WHO/CDC",
                "source": "Replace with source URL or citation",
            }
        ],
    )
    write_table(
        outdir / "clade_rules_template.csv",
        [
            {
                "clade": "example_clade",
                "site": 145,
                "aa": "K",
                "source": "Replace with Nextstrain/Nextclade or literature source",
                "note": "Prefer Nextclade output as the reference clade assignment.",
            }
        ],
    )
    log(f"Template files written to {outdir}")


def run_pipeline(args: argparse.Namespace) -> None:
    outdir = ensure_outdir(Path(args.outdir))
    fasta_records = read_fasta(Path(args.fasta))
    metadata_rows = read_table(Path(args.metadata)) if args.metadata else []
    metadata_index = metadata_by_id(metadata_rows)

    manifest: Dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "safety_scope": "retrospective_public_data_analysis_only",
        "input_fasta": str(Path(args.fasta).resolve()),
        "metadata": str(Path(args.metadata).resolve()) if args.metadata else "",
        "outputs": {},
        "warnings": [],
    }

    prefer_mafft = not args.no_mafft
    normalized_frame_nt: Optional[List[SeqRecord]] = None
    protein_reference_record: Optional[SeqRecord] = None
    if args.normalize:
        log("Normalizing input sequences (clean/filter/dedupe -> reference frame)")
        fasta_clean, fasta_frame, norm_rows, nt_ref, nt_method = normalize_and_frame(
            fasta_records,
            "auto",
            args.reference_id,
            args.min_seq_length,
            args.max_ambiguous_fraction,
            args.collapse_duplicates,
            prefer_mafft,
            args.mafft_threads,
        )
        norm_report_path = outdir / "sequence_normalization_report.csv"
        frame_fasta_path = outdir / "normalized_reference_frame.fasta"
        write_table(norm_report_path, norm_rows)
        write_fasta(frame_fasta_path, fasta_frame)
        manifest["outputs"]["sequence_normalization_report"] = str(norm_report_path)
        manifest["outputs"]["normalized_reference_frame_fasta"] = str(frame_fasta_path)
        manifest["normalization"] = {
            "reference_id": nt_ref.seq_id,
            "reference_length": len(ungapped_clean(nt_ref.sequence)),
            "alignment_method": nt_method,
            "kept": sum(1 for r in norm_rows if r["status"] in {"kept", "reference"}),
            "dropped": sum(1 for r in norm_rows if str(r["status"]).startswith("dropped")),
            "duplicates": sum(1 for r in norm_rows if r["status"] == "duplicate"),
        }
        if nt_method != "mafft" and prefer_mafft:
            manifest["warnings"].append(
                "MAFFT not found on PATH; used pure-Python Needleman-Wunsch fallback "
                "(best for small/closely related sets)."
            )
        fasta_records = fasta_clean
        normalized_frame_nt = fasta_frame

    log("Running sequence QC")
    qc_rows = compute_qc(fasta_records)
    qc_path = outdir / "sequence_qc.csv"
    write_table(qc_path, qc_rows)
    manifest["outputs"]["sequence_qc"] = str(qc_path)
    manifest["qc_summary"] = summarize_qc(qc_rows)

    subtype_rows: List[Dict[str, Any]] = []
    if args.reference_fasta:
        log("Screening query sequences against provided subtype/source references")
        reference_records = read_fasta(Path(args.reference_fasta))
        subtype_rows.extend(
            screen_reference_similarity(
                fasta_records,
                reference_records,
                metadata_index,
                args.expected_subtype_column,
                args.kmer_size,
            )
        )
    if args.blast_results:
        log("Parsing user-provided BLAST result table")
        subtype_rows.extend(summarize_blast_results(Path(args.blast_results)))
    if subtype_rows:
        path = outdir / "subtype_source_screening.csv"
        write_table(path, subtype_rows)
        manifest["outputs"]["subtype_source_screening"] = str(path)

    protein_records = read_fasta(Path(args.protein_fasta)) if args.protein_fasta else []
    if not protein_records and args.use_input_as_protein:
        protein_records = fasta_records

    if args.normalize and protein_records:
        log("Normalizing protein sequences -> reference frame")
        _, protein_frame, protein_norm_rows, protein_reference_record, prot_method = normalize_and_frame(
            protein_records,
            "protein",
            args.protein_reference_id,
            0,
            1.0,
            args.collapse_duplicates,
            prefer_mafft,
            args.mafft_threads,
        )
        protein_report_path = outdir / "protein_normalization_report.csv"
        write_table(protein_report_path, protein_norm_rows)
        manifest["outputs"]["protein_normalization_report"] = str(protein_report_path)
        manifest["protein_normalization"] = {
            "reference_id": protein_reference_record.seq_id,
            "reference_length": len(ungapped_clean(protein_reference_record.sequence)),
            "alignment_method": prot_method,
        }
        protein_records = protein_frame

    nextclade_rows: List[Dict[str, Any]] = []
    rule_clade_rows: List[Dict[str, Any]] = []
    if args.nextclade:
        log("Loading Nextclade clade assignments")
        nextclade_rows = load_nextclade_clades(Path(args.nextclade))
        path = outdir / "nextclade_clades_normalized.csv"
        write_table(path, nextclade_rows)
        manifest["outputs"]["nextclade_clades_normalized"] = str(path)
    if args.clade_rules:
        if not protein_records:
            manifest["warnings"].append("Clade rules were provided, but no protein FASTA was available.")
        else:
            log("Classifying clades with user-provided marker rules")
            rule_rows = read_table(Path(args.clade_rules))
            rule_clade_rows = classify_by_clade_rules(protein_records, rule_rows, args.clade_rule_min_score)
            path = outdir / "rule_based_clades.csv"
            write_table(path, rule_clade_rows)
            manifest["outputs"]["rule_based_clades"] = str(path)
    clade_rows = merge_clade_results(fasta_records, nextclade_rows, rule_clade_rows)
    clade_path = outdir / "clade_assignments.csv"
    write_table(clade_path, clade_rows)
    manifest["outputs"]["clade_assignments"] = str(clade_path)

    tree_svg_path: Optional[Path] = None
    tree_source_records: List[SeqRecord] = []
    tree_note = ""
    if args.aligned_fasta:
        tree_source_records = read_fasta(Path(args.aligned_fasta))
        tree_note = "user_provided_aligned_fasta"
    elif normalized_frame_nt and len(normalized_frame_nt) > 1:
        tree_source_records = normalized_frame_nt
        tree_note = "normalized_reference_frame"
    if tree_source_records:
        log(f"Building UPGMA tree and SVG visualization ({tree_note})")
        tree = build_upgma_tree(tree_source_records)
        if tree:
            newick = node_to_newick(tree) + ";"
            newick_path = outdir / "phylogenetic_tree_upgma.newick"
            newick_path.write_text(newick + "\n", encoding="utf-8")
            clade_by_id = {row["normalized_id"]: row.get("final_clade", "") for row in clade_rows}
            tree_svg_path = outdir / "phylogenetic_tree_upgma.svg"
            render_tree_svg(tree, tree_svg_path, clade_by_id=clade_by_id)
            manifest["outputs"]["phylogenetic_tree_newick"] = str(newick_path)
            manifest["outputs"]["phylogenetic_tree_svg"] = str(tree_svg_path)
            manifest["tree_source"] = tree_note
    else:
        manifest["warnings"].append(
            "No aligned FASTA or --normalize result available; phylogenetic tree generation skipped."
        )

    antigenic_pair_rows: List[Dict[str, Any]] = []
    antigenic_site_rows: List[Dict[str, Any]] = []
    antigenic_coord_rows: List[Dict[str, Any]] = []
    if args.antigenic_sites and args.vaccine_fasta:
        if not protein_records:
            manifest["warnings"].append("Antigenic-site analysis needs protein FASTA; skipped.")
        else:
            log("Computing sequence-based antigenic-site distances")
            sites = load_antigenic_sites(Path(args.antigenic_sites))
            vaccine_records = read_fasta(Path(args.vaccine_fasta))
            if args.normalize and protein_reference_record is not None:
                log("Reframing vaccine/reference strains onto the protein reference")
                vaccine_records, _, _ = frame_records_against(
                    vaccine_records,
                    protein_reference_record,
                    prefer_mafft,
                    args.mafft_threads,
                )
            antigenic_pair_rows, antigenic_site_rows = compute_antigenic_distances(
                protein_records,
                vaccine_records,
                sites,
            )
            antigenic_coord_rows = compute_antigenic_map(protein_records, vaccine_records, sites)
            pair_path = outdir / "antigenic_site_distances.csv"
            site_path = outdir / "antigenic_site_mismatch_matrix.csv"
            coord_path = outdir / "antigenic_site_map_coordinates.csv"
            html_path = outdir / "antigenic_site_map.html"
            write_table(pair_path, antigenic_pair_rows)
            write_table(site_path, antigenic_site_rows)
            write_table(coord_path, antigenic_coord_rows)
            write_antigenic_map_html(html_path, antigenic_coord_rows)
            manifest["outputs"]["antigenic_site_distances"] = str(pair_path)
            manifest["outputs"]["antigenic_site_mismatch_matrix"] = str(site_path)
            manifest["outputs"]["antigenic_site_map_coordinates"] = str(coord_path)
            manifest["outputs"]["antigenic_site_map_html"] = str(html_path)
    elif args.antigenic_sites or args.vaccine_fasta:
        manifest["warnings"].append("Antigenic analysis requires both --antigenic-sites and --vaccine-fasta.")

    resistance_rows: List[Dict[str, Any]] = []
    if args.resistance_markers:
        if not protein_records:
            manifest["warnings"].append("Resistance marker annotation needs protein FASTA; skipped.")
        else:
            log("Annotating known published antiviral resistance markers")
            marker_rows = read_table(Path(args.resistance_markers))
            resistance_rows = annotate_resistance_markers(protein_records, marker_rows, args.protein_gene)
            path = outdir / "known_antiviral_marker_annotations.csv"
            write_table(path, resistance_rows)
            manifest["outputs"]["known_antiviral_marker_annotations"] = str(path)

    selection_site_rows: List[Dict[str, Any]] = []
    if args.aligned_cds:
        log("Computing educational retrospective selection-pressure summaries")
        cds_records = read_fasta(Path(args.aligned_cds))
        selection_site_rows, selection_summary = analyze_selection_pressure(cds_records)
        site_path = outdir / "selection_pressure_sites.csv"
        summary_path = outdir / "selection_pressure_summary.json"
        write_table(site_path, selection_site_rows)
        write_json(summary_path, selection_summary)
        manifest["outputs"]["selection_pressure_sites"] = str(site_path)
        manifest["outputs"]["selection_pressure_summary"] = str(summary_path)
    else:
        manifest["warnings"].append("No aligned CDS FASTA provided; selection-pressure summary skipped.")

    phylo_geo_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []
    if metadata_rows:
        log("Summarizing region/time metadata for phylogeography-style visualization")
        phylo_geo_rows, region_rows = summarize_phylogeography(
            metadata_rows,
            clade_rows,
            args.date_column,
            args.region_column,
            args.time_bin,
        )
        geo_path = outdir / "phylogeography_time_region_clade_counts.csv"
        region_path = outdir / "phylogeography_region_clade_counts.csv"
        write_table(geo_path, phylo_geo_rows)
        write_table(region_path, region_rows)
        manifest["outputs"]["phylogeography_time_region_clade_counts"] = str(geo_path)
        manifest["outputs"]["phylogeography_region_clade_counts"] = str(region_path)

    chart_data = {
        "qc": qc_rows,
        "clades": clade_rows,
        "phylogeography_counts": phylo_geo_rows,
        "selection_sites": selection_site_rows,
        "resistance_markers": resistance_rows,
        "antigenic_map": antigenic_coord_rows,
    }
    dashboard_path = outdir / "influenza_a_analysis_dashboard.html"
    write_dashboard_html(dashboard_path, args.title, manifest, chart_data, tree_svg_path)
    manifest["outputs"]["dashboard_html"] = str(dashboard_path)

    manifest_path = outdir / "analysis_manifest.json"
    write_json(manifest_path, manifest)
    log(f"Done. Dashboard: {dashboard_path}")
    log(f"Manifest: {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrospective Influenza A sequence analysis and visualization toolkit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the analysis pipeline")
    run.add_argument("--fasta", required=True, help="Input nucleotide or aligned sequence FASTA")
    run.add_argument("--protein-fasta", help="Aligned protein FASTA for HA/NA/M2 modules")
    run.add_argument("--use-input-as-protein", action="store_true", help="Treat --fasta as protein FASTA for protein modules")
    run.add_argument("--metadata", help="Metadata CSV/TSV with seqName/id plus date/region/subtype columns")
    run.add_argument("--outdir", default="influenza_analysis_results", help="Output directory")
    run.add_argument("--title", default="Influenza A Retrospective Analysis Dashboard", help="Dashboard title")

    run.add_argument("--reference-fasta", help="Subtype/source reference FASTA for similarity screening")
    run.add_argument("--expected-subtype-column", default="subtype", help="Metadata column with expected subtype")
    run.add_argument("--blast-results", help="CSV/TSV produced by blast_version2.py or another BLAST parser")
    run.add_argument("--kmer-size", type=int, default=15, help="K-mer size for unaligned similarity screening")

    run.add_argument("--nextclade", help="Nextclade CSV/TSV output for clade assignment")
    run.add_argument("--clade-rules", help="CSV/TSV marker-rule table: clade,site,aa")
    run.add_argument("--clade-rule-min-score", type=float, default=0.75, help="Minimum marker-rule score for clade assignment")

    run.add_argument("--aligned-fasta", help="Aligned FASTA for UPGMA tree visualization")
    run.add_argument("--vaccine-fasta", help="Aligned protein FASTA containing vaccine/reference strains")
    run.add_argument("--antigenic-sites", help="CSV/TSV with published antigenic-site positions")
    run.add_argument("--resistance-markers", help="CSV/TSV with known published antiviral marker annotations")
    run.add_argument("--protein-gene", default="", help="Protein gene for marker filtering, e.g. HA, NA, M2")
    run.add_argument("--aligned-cds", help="Aligned coding-sequence FASTA for educational selection analysis")

    run.add_argument("--date-column", help="Metadata date column")
    run.add_argument("--region-column", help="Metadata region/country/location column")
    run.add_argument("--time-bin", choices=["month", "quarter", "year"], default="month", help="Time bin for metadata summaries")

    run.add_argument("--normalize", action="store_true", help="Clean/filter/dedupe and align input onto a reference coordinate frame before positional modules")
    run.add_argument("--reference-id", help="seqName/id within --fasta to use as the nucleotide alignment+coordinate reference (default: longest)")
    run.add_argument("--protein-reference-id", help="seqName/id within the protein set to use as the protein coordinate reference (default: longest)")
    run.add_argument("--min-seq-length", type=int, default=0, help="Drop sequences shorter than this many ungapped residues (0 = off)")
    run.add_argument("--max-ambiguous-fraction", type=float, default=1.0, help="Drop sequences whose ambiguous fraction exceeds this (1.0 = off)")
    run.add_argument("--collapse-duplicates", action="store_true", help="Collapse identical (cleaned, ungapped) sequences, keeping the first")
    run.add_argument("--no-mafft", action="store_true", help="Skip MAFFT even if installed; use the pure-Python Needleman-Wunsch aligner")
    run.add_argument("--mafft-threads", type=int, default=1, help="Threads passed to MAFFT when available")
    run.set_defaults(func=run_pipeline)

    templates = subparsers.add_parser("templates", help="Write CSV templates for metadata, sites, markers, and clade rules")
    templates.add_argument("--outdir", default="influenza_analysis_templates", help="Template output directory")
    templates.set_defaults(func=lambda args: write_templates(Path(args.outdir)))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"[influenza-analysis][ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
