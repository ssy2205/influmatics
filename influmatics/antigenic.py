"""Antigenic-site mutation scanning.

This scanner expects an **amino-acid** mutation table. H3 / H1 antigenic
site definitions are stated in protein coordinates (e.g. H3 site A includes
residue 145). The companion ``mutations`` command emits nucleotide-level
mutations by default, so we explicitly check the ``coordinate_space`` column
of the input TSV and refuse to silently scan NT data. The previous
implementation produced empty results without warning when this happened.
"""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

ALLOW_LEGACY_INPUT = True


class CoordinateSpaceError(ValueError):
    """Raised when a mutation table is in the wrong coordinate space."""


@dataclass(frozen=True)
class AntigenicSiteDefinition:
    numbering: str
    sites: dict[str, list[int]]
    note: str = ""


@dataclass(frozen=True)
class AntigenicHit:
    seq_id: str
    mutation: str
    position: int
    site: str
    numbering: str
    note: str = ""


def mutation_in_sites(position: int, sites: dict[str, list[int]]) -> list[str]:
    """Return antigenic site names that contain a position."""

    return [name for name, positions in sites.items() if position in positions]


def read_antigenic_sites(path: str | Path) -> AntigenicSiteDefinition:
    """Read antigenic site definitions from JSON."""

    with Path(path).open() as handle:
        payload = json.load(handle)
    sites = payload.get("sites")
    if not isinstance(sites, dict):
        raise ValueError("Antigenic site JSON must include a sites object.")
    parsed_sites: dict[str, list[int]] = {}
    for site, positions in sites.items():
        if not isinstance(positions, list):
            raise ValueError(f"Antigenic site positions must be a list: {site}")
        parsed_sites[site] = [int(position) for position in positions]
    return AntigenicSiteDefinition(
        numbering=payload.get("numbering", ""),
        sites=parsed_sites,
        note=payload.get("note", ""),
    )


def read_mutation_rows(path: str | Path) -> list[dict[str, str]]:
    """Read mutation TSV rows with seq_id, mutation, and position columns.

    Verifies amino-acid coordinate space, matching the antigenic-site
    definitions. Falls back to legacy (column-absent) behavior with a
    warning so externally-prepared AA tables keep working.
    """
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        required = {"seq_id", "mutation", "position"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(
                f"Mutation table is missing columns: {','.join(sorted(missing))}"
            )
        rows = list(reader)

    if "coordinate_space" in fieldnames:
        spaces = {row.get("coordinate_space", "").lower() for row in rows} - {""}
        if spaces and spaces != {"aa"}:
            raise CoordinateSpaceError(
                "Antigenic-site scanning requires amino-acid (aa) coordinates. "
                f"Got coordinate_space values: {sorted(spaces)}. Translate the "
                "mutation table to AA coordinates first (e.g. via the "
                "numbering mapper) before running antigenic scan."
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
            "amino-acid coordinates for antigenic-site scanning. If your "
            "input is nucleotide-level, results will be empty and incorrect.",
            RuntimeWarning,
            stacklevel=2,
        )

    return rows


def scan_antigenic_sites(
    mutation_rows: list[dict[str, str]],
    definition: AntigenicSiteDefinition,
) -> list[AntigenicHit]:
    """Scan mutation rows for positions in antigenic sites."""

    hits: list[AntigenicHit] = []
    for row in mutation_rows:
        try:
            position = int(row["position"])
        except (ValueError, KeyError):
            continue
        for site in mutation_in_sites(position, definition.sites):
            hits.append(
                AntigenicHit(
                    seq_id=row["seq_id"],
                    mutation=row["mutation"],
                    position=position,
                    site=site,
                    numbering=definition.numbering,
                    note=definition.note,
                )
            )
    return hits


def antigenic_hits_to_rows(hits: list[AntigenicHit]) -> list[dict[str, object]]:
    """Convert antigenic hits into TSV-friendly dictionaries."""

    return [
        {
            "seq_id": hit.seq_id,
            "mutation": hit.mutation,
            "position": hit.position,
            "site": hit.site,
            "numbering": hit.numbering,
            "note": hit.note,
        }
        for hit in hits
    ]
