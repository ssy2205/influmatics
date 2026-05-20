"""Coordinate and numbering helpers.

NOTE: This module maps **ungapped reference position <-> aligned column
position**. It does NOT translate nucleotide mutations to amino-acid
mutations. NT->AA translation is a separate concern tracked in a
follow-up issue.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NumberingEntry:
    scheme: str
    gene: str
    reference_position: int
    numbering_label: str
    note: str = ""


@dataclass(frozen=True)
class NumberingMapRow:
    scheme: str
    gene: str
    reference_position: int
    numbering_label: str
    alignment_position: int | None
    reference_base: str | None
    note: str = ""

def ungapped_to_aligned_positions(aligned_sequence: str) -> dict[int, int]:
    """Map 1-based ungapped sequence positions to 1-based aligned coordinates."""

    mapping: dict[int, int] = {}
    ungapped_position = 0
    for aligned_position, base in enumerate(aligned_sequence, start=1):
        if base == "-":
            continue
        ungapped_position += 1
        mapping[ungapped_position] = aligned_position
    return mapping


def aligned_to_ungapped_positions(aligned_sequence: str) -> dict[int, int | None]:
    """Map 1-based aligned coordinates to 1-based ungapped positions."""

    mapping: dict[int, int | None] = {}
    ungapped_position = 0
    for aligned_position, base in enumerate(aligned_sequence, start=1):
        if base == "-":
            mapping[aligned_position] = None
            continue
        ungapped_position += 1
        mapping[aligned_position] = ungapped_position
    return mapping


def read_numbering_table(path: str | Path) -> list[NumberingEntry]:
    """Read a numbering table with scheme, gene, reference_position, and numbering_label."""

    entries: list[NumberingEntry] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"scheme", "gene", "reference_position", "numbering_label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Numbering table is missing columns: {','.join(sorted(missing))}")
        for row in reader:
            entries.append(
                NumberingEntry(
                    scheme=row["scheme"],
                    gene=row["gene"],
                    reference_position=int(row["reference_position"]),
                    numbering_label=row["numbering_label"],
                    note=row.get("note", ""),
                )
            )
    return entries


def build_numbering_map(
    aligned_reference: str,
    entries: list[NumberingEntry],
    scheme: str | None = None,
    gene: str | None = None,
) -> list[NumberingMapRow]:
    """Map numbering-table rows onto an aligned reference sequence."""

    ungapped_to_aligned = ungapped_to_aligned_positions(aligned_reference)
    rows: list[NumberingMapRow] = []
    for entry in entries:
        if scheme and entry.scheme != scheme:
            continue
        if gene and entry.gene != gene:
            continue
        alignment_position = ungapped_to_aligned.get(entry.reference_position)
        reference_base = None
        if alignment_position is not None:
            reference_base = aligned_reference[alignment_position - 1].upper()
        rows.append(
            NumberingMapRow(
                scheme=entry.scheme,
                gene=entry.gene,
                reference_position=entry.reference_position,
                numbering_label=entry.numbering_label,
                alignment_position=alignment_position,
                reference_base=reference_base,
                note=entry.note,
            )
        )
    return rows


def numbering_rows_to_tsv(rows: list[NumberingMapRow]) -> list[dict[str, object]]:
    """Convert numbering map rows into TSV-friendly dictionaries."""

    return [
        {
            "scheme": row.scheme,
            "gene": row.gene,
            "reference_position": row.reference_position,
            "numbering_label": row.numbering_label,
            # Use `is not None` so an alignment_position of 0 isn't coerced
            # to "" (defensive: callers might pre-fill 0-based positions),
            # and an empty-string reference_base stays empty rather than
            # being silently dropped.
            "alignment_position": (
                row.alignment_position if row.alignment_position is not None else ""
            ),
            "reference_base": (
                row.reference_base if row.reference_base is not None else ""
            ),
            "note": row.note,
        }
        for row in rows
    ]
