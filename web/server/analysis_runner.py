from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from .schemas import AnalysisOptions, JobStatus


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "web" / "runs"
LEGACY_SCRIPT = REPO_ROOT / "legacy" / "h3n2_ha_analysis.py"

INPUT_FILENAMES = {
    "target": "target.fasta",
    "reference": "reference.fasta",
    "background": "background.fasta",
    "vaccine": "vaccine.fasta",
    "tree_date_metadata": "tree_dates.csv",
    "nextclade_results": "nextclade_results.tsv",
    "tree_outlier_file": "tree_outliers.txt",
}

REQUIRED_INPUTS = {"target"}


@dataclass
class JobRecord:
    run_id: str
    run_dir: Path
    inputs_dir: Path
    results_dir: Path
    log_path: Path
    options: AnalysisOptions
    status: JobStatus = JobStatus.queued
    return_code: Optional[int] = None
    message: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    process: Optional[subprocess.Popen[str]] = None


class AnalysisRunner:
    """Create run folders and execute the legacy CLI in a subprocess."""

    def __init__(
        self,
        runs_root: Path = DEFAULT_RUNS_ROOT,
        repo_root: Path = REPO_ROOT,
        legacy_script: Path = LEGACY_SCRIPT,
    ) -> None:
        self.runs_root = runs_root
        self.repo_root = repo_root
        self.legacy_script = legacy_script
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        file_payloads: Mapping[str, bytes],
        options: AnalysisOptions,
    ) -> JobRecord:
        missing = sorted(name for name in REQUIRED_INPUTS if not file_payloads.get(name))
        if missing:
            raise ValueError(f"Missing required input file(s): {', '.join(missing)}")

        run_id = self._new_run_id()
        run_dir = self.runs_root / run_id
        inputs_dir = run_dir / "inputs"
        results_dir = run_dir / "results"
        log_path = run_dir / "run.log"
        inputs_dir.mkdir(parents=True, exist_ok=False)
        results_dir.mkdir(parents=True, exist_ok=False)
        log_path.write_text("", encoding="utf-8")

        for field_name, payload in file_payloads.items():
            if not payload:
                continue
            filename = INPUT_FILENAMES.get(field_name)
            if filename is None:
                continue
            (inputs_dir / filename).write_bytes(payload)

        job = JobRecord(
            run_id=run_id,
            run_dir=run_dir,
            inputs_dir=inputs_dir,
            results_dir=results_dir,
            log_path=log_path,
            options=options,
        )
        self._write_status(job)
        with self._lock:
            self._jobs[run_id] = job
        return job

    def start_job(self, job: JobRecord) -> None:
        worker = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        worker.start()

    def get_job(self, run_id: str) -> Optional[JobRecord]:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is not None:
            return job
        return self._load_status(run_id)

    def cancel_job(self, run_id: str) -> JobRecord:
        job = self.require_job(run_id)
        if job.process and job.status in {JobStatus.queued, JobStatus.running}:
            job.process.terminate()
            job.status = JobStatus.cancelled
            job.message = "Cancellation requested."
            self._write_status(job)
        return job

    def require_job(self, run_id: str) -> JobRecord:
        job = self.get_job(run_id)
        if job is None:
            raise KeyError(run_id)
        return job

    def build_command(self, job: JobRecord) -> list[str]:
        inputs = job.inputs_dir
        results = job.results_dir
        opts = job.options

        cmd = [
            sys.executable,
            str(self.legacy_script),
            "--target",
            str(inputs / "target.fasta"),
            "--outdir",
            str(results),
            "--tree-method",
            opts.tree_method,
            "--tree-plot-style",
            opts.tree_plot_style,
            "--tree-display-max-tips",
            str(opts.tree_display_max_tips),
            "--tree-display-branch-cap",
            str(opts.tree_display_branch_cap),
            "--max-tree-sequences",
            str(opts.max_tree_sequences),
            "--iqtree-model",
            opts.iqtree_model,
            "--iqtree-threads",
            opts.iqtree_threads,
            "--treetime-outlier-max-passes",
            str(opts.treetime_outlier_max_passes),
            "--clade-method",
            opts.clade_method,
        ]

        optional_file_args = [
            ("reference", "--reference"),
            ("background", "--background"),
            ("vaccine", "--vaccine"),
            ("tree_date_metadata", "--tree-date-metadata"),
            ("nextclade_results", "--nextclade-results"),
            ("tree_outlier_file", "--tree-outlier-file"),
        ]
        for field_name, flag in optional_file_args:
            path = inputs / INPUT_FILENAMES[field_name]
            if path.exists():
                cmd.extend([flag, str(path)])

        scalar_options = [
            (opts.target_date, "--target-date"),
            (opts.iqtree_exe, "--iqtree-exe"),
            (opts.treetime_exe, "--treetime-exe"),
        ]
        for value, flag in scalar_options:
            value = str(value).strip()
            if value:
                cmd.extend([flag, value])

        if opts.iqtree_fast:
            cmd.append("--iqtree-fast")
        if opts.treetime_remove_outliers:
            cmd.append("--treetime-remove-outliers")
        if opts.allow_rule_clade_fallback:
            cmd.append("--allow-rule-clade-fallback")

        return cmd

    def parse_manifest(self, run_id: str) -> dict:
        job = self.require_job(run_id)
        manifest_path = job.results_dir / "run_manifest.json"
        if not manifest_path.exists():
            return {}
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def list_result_files(self, run_id: str) -> list[dict]:
        job = self.require_job(run_id)
        files = []
        for path in sorted(job.results_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(job.results_dir).as_posix()
            files.append(
                {
                    "name": rel,
                    "size": path.stat().st_size,
                    "url": f"/analyses/{run_id}/files/{rel}",
                }
            )
        return files

    def result_file_path(self, run_id: str, file_path: str) -> Path:
        job = self.require_job(run_id)
        resolved = (job.results_dir / file_path).resolve()
        try:
            resolved.relative_to(job.results_dir.resolve())
        except ValueError as exc:
            raise ValueError("Result file path escapes run directory.") from exc
        if not resolved.is_file():
            raise FileNotFoundError(file_path)
        return resolved

    def log_tail(self, run_id: str, max_bytes: int = 12000) -> str:
        job = self.require_job(run_id)
        if not job.log_path.exists():
            return ""
        size = job.log_path.stat().st_size
        with job.log_path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            data = handle.read()
        return data.decode("utf-8", errors="replace")

    def _run_job(self, job: JobRecord) -> None:
        job.status = JobStatus.running
        job.message = "Legacy analysis process started."
        self._write_status(job)
        cmd = self.build_command(job)
        with job.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("$ " + " ".join(cmd) + "\n\n")
            log_file.flush()
            try:
                job.process = subprocess.Popen(
                    cmd,
                    cwd=str(self.repo_root),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                job.return_code = job.process.wait()
            except Exception as exc:  # pragma: no cover - defensive subprocess guard
                job.return_code = -1
                job.status = JobStatus.failed
                job.message = f"Failed to start analysis: {exc}"
                log_file.write(job.message + "\n")
                self._write_status(job)
                return

        if job.status == JobStatus.cancelled:
            job.message = "Analysis cancelled."
        elif job.return_code == 0 and (job.results_dir / "run_manifest.json").exists():
            job.status = JobStatus.completed
            job.message = "Analysis completed."
        else:
            job.status = JobStatus.failed
            job.message = f"Analysis failed with return code {job.return_code}."
        self._write_status(job)

    def _new_run_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"

    def _status_path(self, run_id: str) -> Path:
        return self.runs_root / run_id / "job_status.json"

    def _write_status(self, job: JobRecord) -> None:
        payload = {
            "run_id": job.run_id,
            "status": job.status.value,
            "return_code": job.return_code,
            "message": job.message,
            "created_at": job.created_at,
            "options": job.options.model_dump(),
        }
        (job.run_dir / "job_status.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _load_status(self, run_id: str) -> Optional[JobRecord]:
        status_path = self._status_path(run_id)
        if not status_path.exists():
            return None
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        run_dir = self.runs_root / run_id
        job = JobRecord(
            run_id=run_id,
            run_dir=run_dir,
            inputs_dir=run_dir / "inputs",
            results_dir=run_dir / "results",
            log_path=run_dir / "run.log",
            options=AnalysisOptions(**payload.get("options", {})),
            status=JobStatus(payload.get("status", "failed")),
            return_code=payload.get("return_code"),
            message=payload.get("message", ""),
            created_at=payload.get("created_at", ""),
        )
        with self._lock:
            self._jobs[run_id] = job
        return job


def copy_example_inputs(destination: Path, sources: Iterable[Path]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        if source.exists():
            shutil.copy2(source, destination / source.name)
