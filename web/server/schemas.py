from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class AnalysisOptions(BaseModel):
    tree_method: str = "auto"
    tree_plot_style: str = "figtree"
    tree_display_max_tips: int = 0
    tree_display_branch_cap: float = 0.65
    max_tree_sequences: int = 0
    target_date: str = ""
    iqtree_model: str = "GTR+G"
    iqtree_threads: str = "AUTO"
    iqtree_fast: bool = True
    treetime_remove_outliers: bool = False
    treetime_outlier_max_passes: int = 6
    tree_clade_bar: bool = False
    clade_method: str = "auto"
    allow_rule_clade_fallback: bool = True
    iqtree_exe: str = ""
    treetime_exe: str = ""


class FileInfo(BaseModel):
    name: str
    size: int
    url: str


class AnalysisCreateResponse(BaseModel):
    run_id: str
    status: JobStatus


class AnalysisStatusResponse(BaseModel):
    run_id: str
    status: JobStatus
    return_code: Optional[int] = None
    log_tail: str = ""
    message: str = ""


class AnalysisResultsResponse(BaseModel):
    run_id: str
    status: JobStatus
    manifest: Dict[str, Any] = Field(default_factory=dict)
    files: List[FileInfo] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
