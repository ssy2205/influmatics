"""Alignment wrappers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class AlignmentError(RuntimeError):
    """Raised when an external alignment command cannot complete."""


@dataclass(frozen=True)
class AlignmentResult:
    input_fasta: Path
    output_fasta: Path
    command: tuple[str, ...]
    stderr: str


def build_mafft_command(
    input_fasta: str | Path,
    threads: int = 1,
    auto: bool = True,
    reorder: bool = False,
) -> list[str]:
    """Build a MAFFT command with conservative defaults."""

    if threads < 1:
        raise ValueError("threads must be >= 1")

    command = ["mafft", "--thread", str(threads)]
    if auto:
        command.append("--auto")
    if reorder:
        command.append("--reorder")
    command.append(str(input_fasta))
    return command


def ensure_mafft_available() -> str:
    """Return the MAFFT executable path or raise a clear error."""

    executable = shutil.which("mafft")
    if executable is None:
        raise AlignmentError("MAFFT was not found on PATH. Install mafft before running alignment.")
    return executable


def run_mafft(
    input_fasta: str | Path,
    output_fasta: str | Path,
    threads: int = 1,
    auto: bool = True,
    reorder: bool = False,
    timeout: float | None = None,
) -> AlignmentResult:
    """Run MAFFT and write aligned FASTA.

    The output file is written via a sibling ``.tmp`` path and renamed
    atomically on success, so callers can never pick up a half-written
    alignment if MAFFT crashes or times out partway through.
    """

    ensure_mafft_available()
    input_path = Path(input_fasta)
    if not input_path.exists():
        raise AlignmentError(f"Input FASTA does not exist: {input_path}")
    if not input_path.is_file():
        raise AlignmentError(f"Input FASTA is not a file: {input_path}")
    output_path = Path(output_fasta)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_mafft_command(
        input_path,
        threads=threads,
        auto=auto,
        reorder=reorder,
    )
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with tmp_path.open("w") as output_handle:
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=output_handle,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise AlignmentError(
                    f"MAFFT timed out after {exc.timeout}s: {' '.join(command)}"
                ) from exc
        if completed.returncode != 0:
            raise AlignmentError(
                f"MAFFT failed with exit code {completed.returncode}: {completed.stderr.strip()}"
            )
        tmp_path.replace(output_path)
    except BaseException:
        # Best-effort cleanup of the half-written temp file.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return AlignmentResult(
        input_fasta=input_path,
        output_fasta=output_path,
        command=tuple(command),
        stderr=completed.stderr,
    )
