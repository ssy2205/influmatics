"""Feature generation from amino-acid mutation tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_antigenic_site_json(path: str | Path) -> dict[str, set[int]]:
    """Read an antigenic-site JSON file as site -> positions."""

    with Path(path).open() as handle:
        payload = json.load(handle)
    sites = payload.get("sites")
    if not isinstance(sites, dict):
        raise ValueError("Antigenic site JSON must include a sites object")
    return {
        str(site): {int(position) for position in positions}
        for site, positions in sites.items()
    }


def build_mutation_feature_table(
    mutation_rows: pd.DataFrame,
    antigenic_sites: dict[str, set[int]] | None = None,
) -> pd.DataFrame:
    """Summarize AA mutation rows into one feature row per sequence."""

    required = {"seq_id", "position", "mutation_type"}
    missing = sorted(required.difference(mutation_rows.columns))
    if missing:
        raise ValueError(f"Mutation table is missing columns: {','.join(missing)}")
    if "coordinate_space" in mutation_rows.columns:
        spaces = set(mutation_rows["coordinate_space"].astype(str).str.lower()) - {""}
        if spaces and spaces != {"aa"}:
            raise ValueError(
                "Mutation feature generation requires amino-acid coordinates; "
                f"got coordinate_space values: {sorted(spaces)}"
            )

    frame = mutation_rows.copy()
    frame["position"] = pd.to_numeric(frame["position"], errors="coerce")
    frame = frame.dropna(subset=["seq_id", "position"])
    frame["position"] = frame["position"].astype(int)
    frame["mutation_type"] = frame["mutation_type"].astype(str)

    rows: list[dict[str, object]] = []
    for seq_id, group in frame.groupby("seq_id", sort=True):
        row: dict[str, object] = {
            "seq_id": seq_id,
            "aa_mutation_count": int(len(group)),
            "aa_substitution_count": int((group["mutation_type"] == "substitution").sum()),
            "aa_synonymous_count": int((group["mutation_type"] == "synonymous").sum()),
            "aa_deletion_count": int(group["mutation_type"].str.contains("deletion").sum()),
            "aa_insertion_count": int(group["mutation_type"].str.contains("insertion").sum()),
            "aa_frameshift_count": int((group["mutation_type"] == "frameshift").sum()),
            "aa_nonsense_count": int((group["mutation_type"] == "nonsense").sum()),
            "aa_ambiguous_count": int((group["mutation_type"] == "ambiguous").sum()),
        }
        if antigenic_sites is not None:
            total_hits = 0
            positions = set(group["position"])
            for site, site_positions in sorted(antigenic_sites.items()):
                count = len(positions & site_positions)
                row[f"antigenic_site_{site}_mutation_count"] = count
                total_hits += count
            row["antigenic_site_mutation_count"] = total_hits
        rows.append(row)

    if rows:
        return pd.DataFrame(rows)

    columns = [
        "seq_id",
        "aa_mutation_count",
        "aa_substitution_count",
        "aa_synonymous_count",
        "aa_deletion_count",
        "aa_insertion_count",
        "aa_frameshift_count",
        "aa_nonsense_count",
        "aa_ambiguous_count",
    ]
    if antigenic_sites is not None:
        columns.extend(
            f"antigenic_site_{site}_mutation_count"
            for site in sorted(antigenic_sites)
        )
        columns.append("antigenic_site_mutation_count")
    return pd.DataFrame(columns=columns)


def read_mutation_tsv(path: str | Path) -> pd.DataFrame:
    """Read an AA mutation TSV."""

    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
