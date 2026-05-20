"""Nextclade wrapper and output parsing."""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class NextcladeError(RuntimeError):
    """Raised when Nextclade cannot run or its outputs cannot be parsed."""


@dataclass(frozen=True)
class NextcladeResult:
    output_dir: Path
    command: tuple[str, ...]
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CladeAssignment:
    seq_id: str
    clade: str
    qc_status: str
    dataset: str = ""


def build_nextclade_command(
    input_fasta: str | Path,
    output_dir: str | Path,
    dataset: str | None = None,
) -> list[str]:
    """Build a Nextclade CLI command."""

    command = ["nextclade", "run", "--output-all", str(output_dir)]
    if dataset:
        command.extend(["--dataset-name", dataset])
    command.append(str(input_fasta))
    return command


def ensure_nextclade_available() -> str:
    """Return the Nextclade executable path or raise a clear error."""

    executable = shutil.which("nextclade")
    if executable is None:
        raise NextcladeError(
            "Nextclade was not found on PATH. Install nextclade before clade assignment."
        )
    return executable


def run_nextclade(
    input_fasta: str | Path,
    output_dir: str | Path,
    dataset: str | None = None,
) -> NextcladeResult:
    """Run Nextclade CLI and return its output directory."""

    ensure_nextclade_available()
    input_path = Path(input_fasta)
    if not input_path.exists():
        raise NextcladeError(f"Input FASTA does not exist: {input_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    command = build_nextclade_command(input_path, output_path, dataset=dataset)
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise NextcladeError(
            f"Nextclade failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )
    return NextcladeResult(
        output_dir=output_path,
        command=tuple(command),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def parse_nextclade_tsv(path: str | Path, dataset: str = "") -> list[CladeAssignment]:
    """Parse core clade assignment fields from a Nextclade TSV file."""

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise NextcladeError(f"Nextclade TSV has no header: {path}")
        seq_column = _first_existing_column(reader.fieldnames, ["seqName", "seq_id", "name"])
        clade_column = _first_existing_column(reader.fieldnames, ["clade", "Nextclade_pango"])
        qc_column = _first_existing_column(
            reader.fieldnames,
            ["qc.overallStatus", "qc_status", "qcStatus"],
            required=False,
        )
        assignments: list[CladeAssignment] = []
        for row in reader:
            assignments.append(
                CladeAssignment(
                    seq_id=row.get(seq_column, ""),
                    clade=row.get(clade_column, ""),
                    qc_status=row.get(qc_column, "") if qc_column else "",
                    dataset=dataset,
                )
            )
    return assignments


def clade_assignments_to_rows(assignments: list[CladeAssignment]) -> list[dict[str, object]]:
    """Convert clade assignments into TSV-friendly dictionaries."""

    return [
        {
            "seq_id": assignment.seq_id,
            "clade": assignment.clade,
            "qc_status": assignment.qc_status,
            "dataset": assignment.dataset,
        }
        for assignment in assignments
    ]


def _first_existing_column(
    fieldnames: list[str],
    candidates: list[str],
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    if required:
        raise NextcladeError(f"Nextclade TSV is missing one of: {','.join(candidates)}")
    return None
