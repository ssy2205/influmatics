"""Alignment wrappers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def run_mafft(input_fasta: str | Path, output_fasta: str | Path, threads: int = 1) -> Path:
    """Run MAFFT and write aligned FASTA."""

    if shutil.which("mafft") is None:
        raise RuntimeError("MAFFT was not found on PATH. Install mafft before running alignment.")

    output_path = Path(output_fasta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["mafft", "--thread", str(threads), str(input_fasta)]
    with output_path.open("w") as output_handle:
        subprocess.run(command, check=True, stdout=output_handle)
    return output_path
