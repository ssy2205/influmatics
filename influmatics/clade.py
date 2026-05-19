"""Nextclade wrapper placeholder."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def run_nextclade(input_fasta: str | Path, output_dir: str | Path, dataset: str | None = None) -> Path:
    """Run Nextclade CLI and return its output directory."""

    if shutil.which("nextclade") is None:
        raise RuntimeError("Nextclade was not found on PATH. Install nextclade before running clade assignment.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    command = ["nextclade", "run", "--output-all", str(output_path)]
    if dataset:
        command.extend(["--dataset-name", dataset])
    command.append(str(input_fasta))
    subprocess.run(command, check=True)
    return output_path
