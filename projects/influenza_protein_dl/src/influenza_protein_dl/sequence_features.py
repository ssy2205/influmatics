"""Protein sequence validation and small baseline features."""

from __future__ import annotations

from dataclasses import dataclass

AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ALPHABET)
AMBIGUOUS_AA = set("XBZJUO")
SPECIAL_AA = set("*-")
ACCEPTED_AA = AA_SET | AMBIGUOUS_AA | SPECIAL_AA


@dataclass(frozen=True)
class ProteinSummary:
    """Summary statistics for one protein sequence."""

    length: int
    valid_aa_count: int
    invalid_aa_count: int
    ambiguous_aa_count: int
    stop_count: int
    gap_count: int
    invalid_characters: str


def normalize_protein_sequence(sequence: str) -> str:
    """Normalize protein sequence text for feature generation."""

    return "".join(sequence.split()).upper()


def summarize_protein_sequence(sequence: str) -> ProteinSummary:
    """Return conservative summary stats for a protein sequence."""

    normalized = normalize_protein_sequence(sequence)
    invalid = sorted({aa for aa in normalized if aa not in ACCEPTED_AA})
    return ProteinSummary(
        length=len(normalized),
        valid_aa_count=sum(1 for aa in normalized if aa in AA_SET),
        invalid_aa_count=sum(1 for aa in normalized if aa not in ACCEPTED_AA),
        ambiguous_aa_count=sum(1 for aa in normalized if aa in AMBIGUOUS_AA),
        stop_count=normalized.count("*"),
        gap_count=normalized.count("-"),
        invalid_characters="".join(invalid),
    )


def aa_composition(sequence: str) -> dict[str, float]:
    """Return amino-acid composition over canonical residues."""

    normalized = normalize_protein_sequence(sequence)
    denominator = sum(1 for aa in normalized if aa in AA_SET)
    if denominator == 0:
        return {f"aa_{aa}": 0.0 for aa in AA_ALPHABET}
    return {
        f"aa_{aa}": sum(1 for observed in normalized if observed == aa) / denominator
        for aa in AA_ALPHABET
    }


def kmer_counts(sequence: str, k: int = 3) -> dict[str, int]:
    """Count canonical amino-acid k-mers."""

    if k < 1:
        raise ValueError("k must be >= 1")
    normalized = normalize_protein_sequence(sequence)
    counts: dict[str, int] = {}
    for idx in range(0, max(len(normalized) - k + 1, 0)):
        kmer = normalized[idx : idx + k]
        if all(aa in AA_SET for aa in kmer):
            counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def feature_row(seq_id: str, sequence: str, description: str = "") -> dict[str, object]:
    """Build one manifest/feature row for a protein sequence."""

    summary = summarize_protein_sequence(sequence)
    row: dict[str, object] = {
        "seq_id": seq_id,
        "description": description,
        "length": summary.length,
        "valid_aa_count": summary.valid_aa_count,
        "invalid_aa_count": summary.invalid_aa_count,
        "ambiguous_aa_count": summary.ambiguous_aa_count,
        "stop_count": summary.stop_count,
        "gap_count": summary.gap_count,
        "invalid_characters": summary.invalid_characters,
    }
    row.update(aa_composition(sequence))
    return row
