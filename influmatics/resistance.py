"""Antiviral resistance marker scanning.

This scanner expects an **amino-acid** mutation table. Antiviral marker
labels in the literature (``H274Y``, ``R292K``, ``I38T``...) are stated in
protein coordinates. The companion ``mutations`` command emits
nucleotide-level mutations by default, so we explicitly check the
``coordinate_space`` column of the input TSV and refuse to silently match
against an NT table. The previous implementation produced empty results
without warning when this happened.
"""

from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Mutation TSVs produced before the ``coordinate_space`` column existed are
# allowed (so externally-prepared AA mutation tables still work), but a clear
# warning is emitted. Set this to ``False`` to make missing-column inputs an
# error.
ALLOW_LEGACY_INPUT = True


class CoordinateSpaceError(ValueError):
    """Raised when a mutation table is in the wrong coordinate space."""


def marker_matches(mutation: str, markers: set[str]) -> bool:
    """Check whether a mutation string matches a curated marker."""

    return mutation.upper() in {marker.upper() for marker in markers}


@dataclass(frozen=True)
class ResistanceMarker:
    drug_class: str
    gene: str
    subtype: str
    numbering: str
    mutation: str
    note: str = ""


@dataclass(frozen=True)
class ResistanceHit:
    seq_id: str
    mutation: str
    drug_class: str
    gene: str
    subtype: str
    numbering: str
    note: str = ""


def read_antiviral_markers(path: str | Path) -> list[ResistanceMarker]:
    """Read antiviral marker definitions from TSV."""

    markers: list[ResistanceMarker] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"drug_class", "gene", "subtype", "numbering", "mutation"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Marker table is missing columns: {','.join(sorted(missing))}"
            )
        for row in reader:
            markers.append(
                ResistanceMarker(
                    drug_class=row["drug_class"],
                    gene=row["gene"],
                    subtype=row["subtype"],
                    numbering=row["numbering"],
                    mutation=row["mutation"],
                    note=row.get("note", ""),
                )
            )
    return markers


def read_mutation_rows(path: str | Path) -> list[dict[str, str]]:
    """Read mutation TSV rows produced by the mutations command.

    Verifies that the file is in amino-acid coordinate space. If the
    ``coordinate_space`` column is missing, falls back to legacy behavior
    (with a warning) so externally-prepared AA tables keep working.
    """
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if "seq_id" not in fieldnames or "mutation" not in fieldnames:
            raise ValueError("Mutation table must include seq_id and mutation columns.")

        rows = list(reader)

    if "coordinate_space" in fieldnames:
        spaces = {row.get("coordinate_space", "").lower() for row in rows} - {""}
        if not spaces:
            # Empty file with just headers; let downstream decide.
            return rows
        if spaces != {"aa"}:
            raise CoordinateSpaceError(
                "Antiviral resistance scanning requires amino-acid (aa) "
                f"coordinates. Got coordinate_space values: {sorted(spaces)}. "
                "Translate the mutation table to AA coordinates first "
                "(e.g. via the numbering mapper) before running resistance scan."
            )
    else:
        if not ALLOW_LEGACY_INPUT:
            raise CoordinateSpaceError(
                "Mutation table is missing the 'coordinate_space' column. "
                "Regenerate the table with the current mutations command or "
                "set ALLOW_LEGACY_INPUT=True."
            )
        warnings.warn(
            "Mutation table has no 'coordinate_space' column. Assuming "
            "amino-acid coordinates for resistance scanning. If your input "
            "is nucleotide-level, results will be empty and incorrect.",
            RuntimeWarning,
            stacklevel=2,
        )

    return rows


def scan_resistance_markers(
    mutation_rows: list[dict[str, str]],
    markers: list[ResistanceMarker],
    gene: str | None = None,
    subtype: str | None = None,
) -> list[ResistanceHit]:
    """Scan mutation rows against curated resistance marker definitions.

    Same mutation label can map to multiple markers (e.g. across drug classes
    or alternate amino acids). We use ``defaultdict(list)`` so we do not drop
    overlapping records.
    """
    filtered_markers = [
        marker
        for marker in markers
        if (gene is None or marker.gene == gene)
        and (subtype is None or marker.subtype in {subtype, "any"})
    ]

    marker_lookup: dict[str, list[ResistanceMarker]] = defaultdict(list)
    for marker in filtered_markers:
        marker_lookup[marker.mutation.upper()].append(marker)

    hits: list[ResistanceHit] = []
    for row in mutation_rows:
        for marker in marker_lookup.get(row["mutation"].upper(), ()):
            hits.append(
                ResistanceHit(
                    seq_id=row["seq_id"],
                    mutation=row["mutation"],
                    drug_class=marker.drug_class,
                    gene=marker.gene,
                    subtype=marker.subtype,
                    numbering=marker.numbering,
                    note=marker.note,
                )
            )
    return hits


def resistance_hits_to_rows(hits: list[ResistanceHit]) -> list[dict[str, object]]:
    """Convert resistance hits into TSV-friendly dictionaries."""

    return [
        {
            "seq_id": hit.seq_id,
            "mutation": hit.mutation,
            "drug_class": hit.drug_class,
            "gene": hit.gene,
            "subtype": hit.subtype,
            "numbering": hit.numbering,
            "note": hit.note,
        }
        for hit in hits
    ]
