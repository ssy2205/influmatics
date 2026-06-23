"""Sequence-level quality control."""

from __future__ import annotations

from dataclasses import dataclass

from .io import SeqRecord


DNA_ALPHABET = set("ACGTURYKMSWBDHVN-.?")
GAP_CHARS = set("-.")
AMBIGUOUS_BASES = set("URYKMSWBDHVN?")


@dataclass(frozen=True)
class SequenceQc:
    seq_id: str
    length: int
    ambiguous_bases: int
    ambiguous_fraction: float
    gaps: int
    gap_fraction: float
    gc_fraction: float
    invalid_characters: str
    invalid_count: int
    fail_reasons: str
    pass_qc: bool


def assess_sequence(
    record: SeqRecord,
    min_length: int = 500,
    max_ambiguous_fraction: float = 0.05,
    max_gap_fraction: float = 0.05,
) -> SequenceQc:
    """Assess one nucleotide sequence with conservative defaults."""

    sequence = record.sequence.upper()
    invalid = sorted({base for base in sequence if base not in DNA_ALPHABET})
    ambiguous = sum(1 for base in sequence if base in AMBIGUOUS_BASES)
    gaps = sum(1 for base in sequence if base in GAP_CHARS)
    length = len(sequence)
    ambiguous_fraction = ambiguous / length if length else 1.0
    gap_fraction = gaps / length if length else 1.0
    gc_fraction = sum(1 for base in sequence if base in {"G", "C"}) / length if length else 0.0
    fail_reasons = []
    if length < min_length:
        fail_reasons.append("too_short")
    if ambiguous_fraction > max_ambiguous_fraction:
        fail_reasons.append("too_many_ambiguous_bases")
    if gap_fraction > max_gap_fraction:
        fail_reasons.append("too_many_gaps")
    if invalid:
        fail_reasons.append("invalid_characters")
    pass_qc = not fail_reasons
    return SequenceQc(
        seq_id=record.seq_id,
        length=length,
        ambiguous_bases=ambiguous,
        ambiguous_fraction=ambiguous_fraction,
        gaps=gaps,
        gap_fraction=gap_fraction,
        gc_fraction=gc_fraction,
        invalid_characters="".join(invalid),
        invalid_count=sum(1 for base in sequence if base not in DNA_ALPHABET),
        fail_reasons=",".join(fail_reasons),
        pass_qc=pass_qc,
    )


def assess_sequences(
    records: list[SeqRecord],
    min_length: int = 500,
    max_ambiguous_fraction: float = 0.05,
    max_gap_fraction: float = 0.05,
) -> list[SequenceQc]:
    """Assess multiple sequences."""

    return [
        assess_sequence(
            record,
            min_length=min_length,
            max_ambiguous_fraction=max_ambiguous_fraction,
            max_gap_fraction=max_gap_fraction,
        )
        for record in records
    ]


def qc_to_rows(results: list[SequenceQc]) -> list[dict[str, object]]:
    """Convert QC dataclasses into TSV-friendly rows."""

    return [
        {
            "seq_id": result.seq_id,
            "length": result.length,
            "ambiguous_bases": result.ambiguous_bases,
            "ambiguous_fraction": f"{result.ambiguous_fraction:.6f}",
            "gaps": result.gaps,
            "gap_fraction": f"{result.gap_fraction:.6f}",
            "gc_fraction": f"{result.gc_fraction:.6f}",
            "invalid_characters": result.invalid_characters,
            "invalid_count": result.invalid_count,
            "fail_reasons": result.fail_reasons,
            "pass_qc": result.pass_qc,
        }
        for result in results
    ]
