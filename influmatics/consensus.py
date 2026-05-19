"""ONT consensus pipeline wrapper."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_legacy_ont_pipeline(
    reads: str | Path | None,
    reference: str | Path,
    outdir: str | Path,
    script_path: str | Path = "legacy/original_scripts/auto_pipeline.sh",
    indir: str | Path | None = None,
    threads: int = 16,
) -> Path:
    """Run the preserved shell pipeline from Python."""

    if reads and indir:
        raise ValueError("Provide either reads or indir, not both.")
    if not reads and not indir:
        raise ValueError("Provide reads or indir.")

    command = [
        "bash",
        str(script_path),
        "--ref",
        str(reference),
        "--outdir",
        str(outdir),
        "--threads",
        str(threads),
    ]
    if reads:
        command.extend(["--reads", str(reads)])
    if indir:
        command.extend(["--indir", str(indir)])
    subprocess.run(command, check=True)
    return Path(outdir)
