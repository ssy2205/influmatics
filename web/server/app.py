from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .analysis_runner import AnalysisRunner
from .schemas import (
    AnalysisCreateResponse,
    AnalysisOptions,
    AnalysisResultsResponse,
    AnalysisStatusResponse,
    FileInfo,
    JobStatus,
)


app = FastAPI(title="Influmatics Web API", version="0.1.0")
runner = AnalysisRunner()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def api_root() -> dict[str, str]:
    return {
        "name": "Influmatics Web API",
        "status": "ok",
        "frontend": "http://127.0.0.1:5173/",
        "docs": "http://127.0.0.1:8000/docs",
    }


@app.post("/analyses", response_model=AnalysisCreateResponse)
async def create_analysis(
    target: UploadFile = File(...),
    reference: UploadFile = File(...),
    background: Optional[UploadFile] = File(None),
    vaccine: Optional[UploadFile] = File(None),
    tree_date_metadata: Optional[UploadFile] = File(None),
    nextclade_results: Optional[UploadFile] = File(None),
    tree_outlier_file: Optional[UploadFile] = File(None),
    tree_method: str = Form("auto"),
    tree_plot_style: str = Form("figtree"),
    tree_display_max_tips: int = Form(0),
    tree_display_branch_cap: float = Form(0.65),
    max_tree_sequences: int = Form(0),
    target_date: str = Form(""),
    iqtree_model: str = Form("GTR+G"),
    iqtree_threads: str = Form("AUTO"),
    iqtree_fast: bool = Form(True),
    treetime_remove_outliers: bool = Form(False),
    treetime_outlier_max_passes: int = Form(6),
    tree_clade_bar: bool = Form(False),
    clade_method: str = Form("auto"),
    allow_rule_clade_fallback: bool = Form(True),
    iqtree_exe: str = Form(""),
    treetime_exe: str = Form(""),
) -> AnalysisCreateResponse:
    file_payloads = {
        "target": await target.read(),
        "reference": await reference.read(),
        "background": await _read_optional_upload(background),
        "vaccine": await _read_optional_upload(vaccine),
        "tree_date_metadata": await _read_optional_upload(tree_date_metadata),
        "nextclade_results": await _read_optional_upload(nextclade_results),
        "tree_outlier_file": await _read_optional_upload(tree_outlier_file),
    }
    options = AnalysisOptions(
        tree_method=tree_method,
        tree_plot_style=tree_plot_style,
        tree_display_max_tips=tree_display_max_tips,
        tree_display_branch_cap=tree_display_branch_cap,
        max_tree_sequences=max_tree_sequences,
        target_date=target_date,
        iqtree_model=iqtree_model,
        iqtree_threads=iqtree_threads,
        iqtree_fast=iqtree_fast,
        treetime_remove_outliers=treetime_remove_outliers,
        treetime_outlier_max_passes=treetime_outlier_max_passes,
        tree_clade_bar=tree_clade_bar,
        clade_method=clade_method,
        allow_rule_clade_fallback=allow_rule_clade_fallback,
        iqtree_exe=iqtree_exe,
        treetime_exe=treetime_exe,
    )
    try:
        job = runner.create_job(file_payloads, options)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    runner.start_job(job)
    return AnalysisCreateResponse(run_id=job.run_id, status=JobStatus.queued)


@app.get("/analyses/{run_id}/status", response_model=AnalysisStatusResponse)
def analysis_status(run_id: str) -> AnalysisStatusResponse:
    job = _require_job(run_id)
    return AnalysisStatusResponse(
        run_id=job.run_id,
        status=job.status,
        return_code=job.return_code,
        log_tail=runner.log_tail(run_id),
        message=job.message,
    )


@app.get("/analyses/{run_id}/results", response_model=AnalysisResultsResponse)
def analysis_results(run_id: str) -> AnalysisResultsResponse:
    job = _require_job(run_id)
    manifest = runner.parse_manifest(run_id)
    files = [FileInfo(**item) for item in runner.list_result_files(run_id)]
    warnings = []
    if job.status == JobStatus.failed:
        warnings.append(job.message or "Analysis failed.")
    return AnalysisResultsResponse(
        run_id=job.run_id,
        status=job.status,
        manifest=manifest,
        files=files,
        warnings=warnings,
    )


@app.get("/analyses/{run_id}/files/{file_path:path}")
def analysis_file(run_id: str, file_path: str) -> FileResponse:
    _require_job(run_id)
    try:
        path = runner.result_file_path(run_id, file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Result file not found.") from exc
    return FileResponse(path, filename=path.name)


@app.post("/analyses/{run_id}/cancel", response_model=AnalysisStatusResponse)
def cancel_analysis(run_id: str) -> AnalysisStatusResponse:
    try:
        job = runner.cancel_job(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    return AnalysisStatusResponse(
        run_id=job.run_id,
        status=job.status,
        return_code=job.return_code,
        log_tail=runner.log_tail(run_id),
        message=job.message,
    )


async def _read_optional_upload(upload: Optional[UploadFile]) -> bytes:
    if upload is None or not upload.filename:
        return b""
    return await upload.read()


def _require_job(run_id: str):
    try:
        return runner.require_job(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
