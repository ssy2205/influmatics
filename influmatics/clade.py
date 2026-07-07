"""Nextclade wrapper and output parsing."""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# A Nextclade amino-acid change token is "GENE:<ref?><pos><alt>", e.g.
# "HA1:N145K" (substitution) or "HA1:144-" / "HA1:K144-" (deletion). The
# reference residue is optional because Nextclade omits it for some deletion
# encodings; the alt part may be a residue, a stop ("*"), or a gap ("-").
_AA_CHANGE_RE = re.compile(r"^([A-Za-z*]?)(\d+)([A-Za-z*-]*)$")


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


@dataclass(frozen=True)
class AaMutation:
    """An amino-acid change parsed out of a Nextclade TSV.

    ``mutation`` is the gene-stripped label (``N145K``) so it lines up with
    antigenic-site / antiviral-marker definitions, while ``gene`` is kept
    separately for per-gene filtering. ``coordinate_space`` is always ``aa``
    so the antigenic and resistance scanners accept the table directly.
    """

    seq_id: str
    gene: str
    position: int
    mutation: str
    mutation_type: str
    coordinate_space: str = "aa"


def build_nextclade_command(
    input_fasta: str | Path,
    output_dir: str | Path,
    dataset: str | None = None,
    input_dataset: str | Path | None = None,
) -> list[str]:
    """Build a Nextclade CLI command.

    ``dataset`` is a Nextclade-managed dataset name (e.g. ``flu_h3n2_ha``).
    ``input_dataset`` is a path to a locally-downloaded dataset directory,
    which is the normal mode for influenza pipelines that ship their own
    reference data. The two are mutually exclusive.
    """

    if dataset and input_dataset:
        raise ValueError(
            "build_nextclade_command: pass only one of dataset (name) or "
            "input_dataset (path), not both."
        )

    command = ["nextclade", "run", "--output-all", str(output_dir)]
    if dataset:
        command.extend(["--dataset-name", dataset])
    if input_dataset:
        command.extend(["--input-dataset", str(input_dataset)])
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
    input_dataset: str | Path | None = None,
    timeout: float | None = None,
) -> NextcladeResult:
    """Run Nextclade CLI and return its output directory."""

    ensure_nextclade_available()
    input_path = Path(input_fasta)
    if not input_path.exists():
        raise NextcladeError(f"Input FASTA does not exist: {input_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    command = build_nextclade_command(
        input_path,
        output_path,
        dataset=dataset,
        input_dataset=input_dataset,
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise NextcladeError(
            f"Nextclade timed out after {exc.timeout}s: {' '.join(command)}"
        ) from exc
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
        # Influenza datasets emit clade / short_clade / subclade. SARS-CoV-2's
        # Nextclade_pango column was previously listed here, which is wrong for
        # this project (flu workflow) -- drop it to avoid silently picking up
        # an unrelated assignment.
        clade_column = _first_existing_column(
            reader.fieldnames, ["clade", "short_clade", "subclade"]
        )
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


def _parse_aa_change_token(token: str) -> tuple[str, int, str] | None:
    """Split a Nextclade AA token into (gene, position, gene-stripped label).

    Returns ``None`` for tokens that don't carry a gene prefix and position,
    so malformed or empty fragments are skipped rather than crashing the run.
    """

    token = token.strip()
    if not token or ":" not in token:
        return None
    gene, change = token.split(":", 1)
    gene = gene.strip()
    change = change.strip()
    match = _AA_CHANGE_RE.match(change)
    if not gene or match is None:
        return None
    position = int(match.group(2))
    return gene, position, change


def parse_nextclade_aa_mutations(
    path: str | Path,
    include_deletions: bool = True,
) -> list[AaMutation]:
    """Parse per-sequence amino-acid changes from a Nextclade TSV.

    Reads the ``aaSubstitutions`` column (and ``aaDeletions`` when
    ``include_deletions`` is true) and expands the comma-separated tokens
    into one :class:`AaMutation` per change. The result is ready to feed the
    antigenic-site and antiviral-resistance scanners.
    """

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise NextcladeError(f"Nextclade TSV has no header: {path}")
        seq_column = _first_existing_column(fieldnames, ["seqName", "seq_id", "name"])
        sub_column = _first_existing_column(
            fieldnames, ["aaSubstitutions", "aa_substitutions"], required=False
        )
        del_column = _first_existing_column(
            fieldnames, ["aaDeletions", "aa_deletions"], required=False
        )
        if sub_column is None and del_column is None:
            raise NextcladeError(
                "Nextclade TSV is missing amino-acid change columns "
                "(aaSubstitutions / aaDeletions). Re-run Nextclade with a "
                "dataset that emits amino-acid annotations."
            )

        sources: list[tuple[str | None, str]] = [(sub_column, "substitution")]
        if include_deletions:
            sources.append((del_column, "deletion"))

        mutations: list[AaMutation] = []
        for row in reader:
            seq_id = row.get(seq_column, "")
            for column, mutation_type in sources:
                if column is None:
                    continue
                cell = (row.get(column) or "").strip()
                if not cell:
                    continue
                for token in cell.split(","):
                    parsed = _parse_aa_change_token(token)
                    if parsed is None:
                        continue
                    gene, position, label = parsed
                    mutations.append(
                        AaMutation(
                            seq_id=seq_id,
                            gene=gene,
                            position=position,
                            mutation=label,
                            mutation_type=mutation_type,
                        )
                    )
    return mutations


def aa_mutations_to_rows(mutations: list[AaMutation]) -> list[dict[str, object]]:
    """Convert AA mutations into scanner-ready TSV rows."""

    return [
        {
            "seq_id": mutation.seq_id,
            "gene": mutation.gene,
            "position": mutation.position,
            "mutation": mutation.mutation,
            "mutation_type": mutation.mutation_type,
            "coordinate_space": mutation.coordinate_space,
        }
        for mutation in mutations
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
