from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKGROUND_SETS_ROOT = REPO_ROOT / "data" / "background_sets"
DEFAULT_BACKGROUND_DATASET_ID = "h3n2_ha_demo_reference"


@dataclass(frozen=True)
class BackgroundDataset:
    id: str
    label: str
    version: str
    root: Path
    manifest_path: Path
    background_fasta: Path
    metadata_csv: Optional[Path] = None
    tree_dates_csv: Optional[Path] = None
    outlier_file: Optional[Path] = None
    description: str = ""
    sequence_count: int = 0
    date_range: tuple[str, str] = ("", "")
    source_policy: str = ""
    sources: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "version": self.version,
            "description": self.description,
            "sequence_count": self.sequence_count,
            "date_range": list(self.date_range),
            "source_policy": self.source_policy,
            "sources": list(self.sources),
        }


class BackgroundDatasetRegistry:
    def __init__(self, root: Path = BACKGROUND_SETS_ROOT) -> None:
        self.root = root

    def list(self) -> list[BackgroundDataset]:
        datasets = []
        if not self.root.exists():
            return datasets
        for manifest_path in sorted(self.root.glob("*/*/manifest.json")):
            try:
                datasets.append(self._load_manifest(manifest_path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(datasets, key=lambda item: (item.id, item.version))

    def get(self, dataset_id: str) -> BackgroundDataset:
        requested = (dataset_id or DEFAULT_BACKGROUND_DATASET_ID).strip()
        matches = [dataset for dataset in self.list() if dataset.id == requested]
        if not matches:
            raise KeyError(requested)
        return matches[-1]

    def _load_manifest(self, manifest_path: Path) -> BackgroundDataset:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parent
        dataset_id = str(payload.get("id") or manifest_path.parents[1].name).strip()
        label = str(payload.get("label") or dataset_id).strip()
        version = str(payload.get("version") or root.name).strip()
        background_fasta = self._resolve_required(root, payload, "background_fasta")
        metadata_csv = self._resolve_optional(root, payload, "metadata_csv")
        tree_dates_csv = self._resolve_optional(root, payload, "tree_dates_csv")
        outlier_file = self._resolve_optional(root, payload, "outlier_file")
        date_range = payload.get("date_range") or ["", ""]
        if not isinstance(date_range, list) or len(date_range) != 2:
            date_range = ["", ""]
        sources = payload.get("sources") or []
        if not isinstance(sources, list):
            sources = []
        return BackgroundDataset(
            id=dataset_id,
            label=label,
            version=version,
            root=root,
            manifest_path=manifest_path,
            background_fasta=background_fasta,
            metadata_csv=metadata_csv,
            tree_dates_csv=tree_dates_csv,
            outlier_file=outlier_file,
            description=str(payload.get("description") or ""),
            sequence_count=int(payload.get("sequence_count") or 0),
            date_range=(str(date_range[0] or ""), str(date_range[1] or "")),
            source_policy=str(payload.get("source_policy") or ""),
            sources=tuple(str(source) for source in sources),
        )

    def _resolve_required(self, root: Path, payload: dict[str, Any], key: str) -> Path:
        value = payload.get(key)
        if not value:
            raise ValueError(f"Dataset manifest missing required key: {key}")
        path = (root / str(value)).resolve()
        if not path.is_file():
            raise ValueError(f"Dataset file not found: {path}")
        return path

    def _resolve_optional(self, root: Path, payload: dict[str, Any], key: str) -> Optional[Path]:
        value = payload.get(key)
        if not value:
            return None
        path = (root / str(value)).resolve()
        return path if path.is_file() else None


def read_dataset_dates(metadata_csv: Path) -> list[tuple[str, str]]:
    rows = []
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = _first_value(row, ["name", "seqName", "strain", "id", "sample"])
            date = _first_value(row, ["date", "collection_date", "year"])
            if name and date:
                rows.append((name, date))
    return rows


def read_fasta_ids(path: Path) -> list[str]:
    ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                name = line[1:].strip().split()[0]
                if name:
                    ids.append(name)
    return ids


def write_tree_dates(
    output_path: Path,
    background_rows: Iterable[tuple[str, str]],
    target_ids: Iterable[str],
    target_date: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set[str] = set()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["name", "date"])
        for name, date in background_rows:
            if not name or not date or name in seen:
                continue
            writer.writerow([name, date])
            seen.add(name)
            count += 1
        if target_date:
            for name in target_ids:
                if not name or name in seen:
                    continue
                writer.writerow([name, target_date])
                seen.add(name)
                count += 1
    return count


def _first_value(row: dict[str, str], keys: list[str]) -> str:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower(), "")
        if value:
            return str(value).strip()
    return ""
