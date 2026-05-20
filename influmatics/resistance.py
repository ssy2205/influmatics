"""Antiviral resistance marker scanning."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


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
            raise ValueError(f"Marker table is missing columns: {','.join(sorted(missing))}")
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
    """Read mutation TSV rows produced by the mutations command."""

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "seq_id" not in reader.fieldnames or "mutation" not in reader.fieldnames:
            raise ValueError("Mutation table must include seq_id and mutation columns.")
        return list(reader)


def scan_resistance_markers(
    mutation_rows: list[dict[str, str]],
    markers: list[ResistanceMarker],
    gene: str | None = None,
    subtype: str | None = None,
) -> list[ResistanceHit]:
    """Scan mutation rows against curated resistance marker definitions."""

    filtered_markers = [
        marker
        for marker in markers
        if (gene is None or marker.gene == gene)
        and (subtype is None or marker.subtype in {subtype, "any"})
    ]
    marker_lookup = {marker.mutation.upper(): marker for marker in filtered_markers}
    hits: list[ResistanceHit] = []
    for row in mutation_rows:
        marker = marker_lookup.get(row["mutation"].upper())
        if marker is None:
            continue
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
