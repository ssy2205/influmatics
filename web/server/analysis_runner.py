from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from .datasets import (
    DEFAULT_BACKGROUND_DATASET_ID,
    BackgroundDatasetRegistry,
    read_dataset_dates,
    read_fasta_ids,
    write_tree_dates,
)
from .gcs_store import GCSRunStore
from .schemas import AnalysisOptions, JobStatus


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "web" / "runs"
LEGACY_SCRIPT = REPO_ROOT / "legacy" / "h3n2_ha_analysis.py"
DEFAULT_REFERENCE_FASTA = REPO_ROOT / "data" / "references" / "A_Aichi_1968_H3N2_HA.fasta"
DEFAULT_NEXTCLADE_DATASET_ENV = "INFLUMATICS_NEXTCLADE_DATASET"

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
        runs_root: Optional[Path] = None,
        repo_root: Path = REPO_ROOT,
        legacy_script: Path = LEGACY_SCRIPT,
        default_reference_fasta: Path = DEFAULT_REFERENCE_FASTA,
        dataset_registry: Optional[BackgroundDatasetRegistry] = None,
    ) -> None:
        self.runs_root = runs_root or default_runs_root()
        self.repo_root = repo_root
        self.legacy_script = legacy_script
        self.default_reference_fasta = default_reference_fasta
        self.dataset_registry = dataset_registry or BackgroundDatasetRegistry.for_repo(repo_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.store = GCSRunStore.from_env()
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
        reference_path = inputs_dir / INPUT_FILENAMES["reference"]
        if not reference_path.exists():
            if not self.default_reference_fasta.exists():
                raise ValueError(f"Default reference FASTA is missing: {self.default_reference_fasta}")
            shutil.copy2(self.default_reference_fasta, reference_path)
        input_manifest = self._prepare_builtin_dataset(inputs_dir, options)
        self._validate_treetime_inputs(inputs_dir, options)

        job = JobRecord(
            run_id=run_id,
            run_dir=run_dir,
            inputs_dir=inputs_dir,
            results_dir=results_dir,
            log_path=log_path,
            options=options,
        )
        if input_manifest:
            (inputs_dir / "input_manifest.json").write_text(
                json.dumps(input_manifest, indent=2),
                encoding="utf-8",
            )
        self._write_status(job)
        self._sync_inputs(job)
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
            self._sync_run(job)
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

        nextclade_dataset = (
            opts.nextclade_dataset
            or os.getenv(DEFAULT_NEXTCLADE_DATASET_ENV, "")
        ).strip()
        nextclade_results_path = inputs / INPUT_FILENAMES["nextclade_results"]
        if (
            nextclade_dataset
            and opts.clade_method in {"auto", "nextclade"}
            and not nextclade_results_path.exists()
        ):
            cmd.extend(["--nextclade-dataset", nextclade_dataset])

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
        if opts.tree_clade_bar:
            cmd.append("--tree-clade-bar")
        if opts.allow_rule_clade_fallback:
            cmd.append("--allow-rule-clade-fallback")

        return cmd

    def _validate_treetime_inputs(
        self,
        inputs_dir: Path,
        options: AnalysisOptions,
    ) -> None:
        if options.tree_method != "iqtree-treetime":
            return
        tree_dates_path = inputs_dir / INPUT_FILENAMES["tree_date_metadata"]
        if not tree_dates_path.exists():
            raise ValueError(
                "IQ-TREE + TreeTime requires dated tips. Choose a background "
                "preset with metadata, enter the target collection date, or "
                "upload Tree date metadata."
            )
        dated_rows = read_dataset_dates(tree_dates_path)
        if len(dated_rows) < 3:
            raise ValueError(
                "IQ-TREE + TreeTime requires at least 3 dated tips; the current "
                f"inputs provide {len(dated_rows)}. Use a full background preset "
                "or upload Tree date metadata before running TreeTime."
            )

    def parse_manifest(self, run_id: str) -> dict:
        job = self.require_job(run_id)
        manifest_path = job.results_dir / "run_manifest.json"
        if not manifest_path.exists():
            if not self.store:
                return {}
            manifest = self.store.download_json(run_id, "results/run_manifest.json")
            return manifest or {}
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def list_result_files(self, run_id: str) -> list[dict]:
        job = self.require_job(run_id)
        if self.store:
            files = self.store.list_result_files(run_id)
            if files:
                return files

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
        if not resolved.is_file() and self.store:
            self.store.download_file(run_id, f"results/{file_path}", resolved)
        if not resolved.is_file():
            raise FileNotFoundError(file_path)
        return resolved

    def log_tail(self, run_id: str, max_bytes: int = 12000) -> str:
        job = self.require_job(run_id)
        if job.log_path.exists():
            size = job.log_path.stat().st_size
            with job.log_path.open("rb") as handle:
                if size > max_bytes:
                    handle.seek(size - max_bytes)
                data = handle.read()
            return data.decode("utf-8", errors="replace")
        if not self.store:
            return ""
        text = self.store.download_text(run_id, "run.log") or ""
        data = text.encode("utf-8")[-max_bytes:]
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
                self._sync_run(job)
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
        self._sync_run(job)

    def _new_run_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"

    def _prepare_builtin_dataset(
        self,
        inputs_dir: Path,
        options: AnalysisOptions,
    ) -> dict:
        dataset_id = (options.background_dataset or self.dataset_registry.default_dataset_id()).strip()
        if not dataset_id:
            return {}
        try:
            dataset = self.dataset_registry.get(dataset_id)
        except KeyError:
            datasets = self.dataset_registry.list()
            if dataset_id == DEFAULT_BACKGROUND_DATASET_ID and not datasets:
                return {}
            available = ", ".join(item.id for item in datasets) or "none"
            raise ValueError(
                f"Unknown background dataset '{dataset_id}'. Available datasets: {available}"
            ) from None

        manifest = {
            "background_dataset": dataset.public_dict(),
            "auto_prepared_inputs": [],
        }
        background_path = inputs_dir / INPUT_FILENAMES["background"]
        if not background_path.exists():
            shutil.copy2(dataset.background_fasta, background_path)
            manifest["auto_prepared_inputs"].append("background")

        tree_dates_path = inputs_dir / INPUT_FILENAMES["tree_date_metadata"]
        if not tree_dates_path.exists():
            background_rows = []
            if dataset.tree_dates_csv:
                background_rows = read_dataset_dates(dataset.tree_dates_csv)
            elif dataset.metadata_csv:
                background_rows = read_dataset_dates(dataset.metadata_csv)
            if background_rows or options.target_date:
                count = write_tree_dates(
                    tree_dates_path,
                    background_rows=background_rows,
                    target_ids=read_fasta_ids(inputs_dir / INPUT_FILENAMES["target"]),
                    target_date=options.target_date.strip(),
                )
                manifest["auto_prepared_inputs"].append("tree_date_metadata")
                manifest["tree_date_rows"] = count

        outlier_path = inputs_dir / INPUT_FILENAMES["tree_outlier_file"]
        if not outlier_path.exists() and dataset.outlier_file:
            shutil.copy2(dataset.outlier_file, outlier_path)
            manifest["auto_prepared_inputs"].append("tree_outlier_file")

        return manifest

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
        if self.store:
            self.store.upload_json(payload, job.run_id, "job_status.json")

    def _load_status(self, run_id: str) -> Optional[JobRecord]:
        status_path = self._status_path(run_id)
        run_dir = self.runs_root / run_id
        if status_path.exists():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        elif self.store:
            payload = self.store.download_json(run_id, "job_status.json")
            if payload is None:
                return None
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "job_status.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        else:
            return None
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

    def _sync_inputs(self, job: JobRecord) -> None:
        if not self.store:
            return
        for path in sorted(job.inputs_dir.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(job.run_dir).as_posix()
            self.store.upload_file(path, job.run_id, relative_path)

    def _sync_run(self, job: JobRecord) -> None:
        if self.store:
            self.store.upload_run_dir(job.run_dir, job.run_id)


def copy_example_inputs(destination: Path, sources: Iterable[Path]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        if source.exists():
            shutil.copy2(source, destination / source.name)


def default_runs_root() -> Path:
    configured = os.getenv("INFLUMATICS_RUNS_ROOT", "").strip()
    if configured:
        return Path(configured)
    railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if railway_volume:
        return Path(railway_volume) / "runs"
    return DEFAULT_RUNS_ROOT
