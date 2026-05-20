"""Antigenic-site mutation scanning."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


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
    """Read mutation TSV rows with seq_id, mutation, and position columns."""

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"seq_id", "mutation", "position"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Mutation table is missing columns: {','.join(sorted(missing))}")
        return list(reader)


def scan_antigenic_sites(
    mutation_rows: list[dict[str, str]],
    definition: AntigenicSiteDefinition,
) -> list[AntigenicHit]:
    """Scan mutation rows for positions in antigenic sites."""

    hits: list[AntigenicHit] = []
    for row in mutation_rows:
        try:
            position = int(row["position"])
        except ValueError:
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
