import json
import sys

from web.server.datasets import BackgroundDatasetRegistry
from web.server.analysis_runner import AnalysisRunner
from web.server.schemas import AnalysisOptions, JobStatus


def test_create_job_makes_run_structure(tmp_path):
    runner = AnalysisRunner(runs_root=tmp_path / "runs", repo_root=tmp_path)

    job = runner.create_job(
        {
            "target": b">target\nAAAA\n",
            "reference": b">reference\nAAAA\n",
            "background": b">background\nAAAA\n",
        },
        AnalysisOptions(),
    )

    assert job.status == JobStatus.queued
    assert job.inputs_dir.is_dir()
    assert job.results_dir.is_dir()
    assert job.log_path.is_file()
    assert (job.inputs_dir / "target.fasta").read_bytes() == b">target\nAAAA\n"
    assert (job.inputs_dir / "reference.fasta").is_file()
    assert (job.inputs_dir / "background.fasta").is_file()


def test_build_command_preserves_legacy_cli_contract(tmp_path):
    script = tmp_path / "legacy" / "h3n2_ha_analysis.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n")
    default_reference = tmp_path / "default_reference.fasta"
    default_reference.write_text(">default\nAAAA\n")
    runner = AnalysisRunner(
        runs_root=tmp_path / "runs",
        repo_root=tmp_path,
        legacy_script=script,
        default_reference_fasta=default_reference,
    )
    job = runner.create_job(
        {
            "target": b">target\nAAAA\n",
            "tree_date_metadata": b"name,date\nx,2023\n",
            "nextclade_results": b"seqName\tclade\nx\t3C\n",
        },
        AnalysisOptions(
            tree_method="iqtree-treetime",
            target_date="2023",
            treetime_remove_outliers=True,
            tree_clade_bar=True,
            iqtree_fast=True,
        ),
    )

    cmd = runner.build_command(job)

    assert cmd[0] == sys.executable
    assert str(script) in cmd
    assert "--target" in cmd
    assert str(job.inputs_dir / "target.fasta") in cmd
    assert "--reference" in cmd
    assert str(job.inputs_dir / "reference.fasta") in cmd
    assert (job.inputs_dir / "reference.fasta").read_text() == ">default\nAAAA\n"
    assert "--outdir" in cmd
    assert str(job.results_dir) in cmd
    assert "--tree-date-metadata" in cmd
    assert "--nextclade-results" in cmd
    assert "--target-date" in cmd
    assert "--treetime-remove-outliers" in cmd
    assert "--tree-clade-bar" in cmd
    assert "--iqtree-fast" in cmd


def test_build_command_adds_custom_reference_only_when_uploaded(tmp_path):
    script = tmp_path / "legacy" / "h3n2_ha_analysis.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n")
    runner = AnalysisRunner(
        runs_root=tmp_path / "runs",
        repo_root=tmp_path,
        legacy_script=script,
    )
    job = runner.create_job(
        {
            "target": b">target\nAAAA\n",
            "reference": b">reference\nAAAA\n",
            "vaccine": b">vaccine\nAAAA\n",
        },
        AnalysisOptions(),
    )

    cmd = runner.build_command(job)

    assert "--reference" in cmd
    assert str(job.inputs_dir / "reference.fasta") in cmd
    assert "--vaccine" in cmd
    assert str(job.inputs_dir / "vaccine.fasta") in cmd


def test_create_job_autoprepares_builtin_background_and_tree_dates(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "data" / "background_sets" / "demo" / "v1"
    dataset_root.mkdir(parents=True)
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "id": "demo",
                "label": "Demo",
                "version": "v1",
                "background_fasta": "background.fasta",
                "metadata_csv": "metadata.csv",
            }
        )
    )
    (dataset_root / "background.fasta").write_text(">bg\nAAAA\n")
    (dataset_root / "metadata.csv").write_text("name,date\nbg,1968\n")
    default_reference = tmp_path / "default_reference.fasta"
    default_reference.write_text(">default\nAAAA\n")
    runner = AnalysisRunner(
        runs_root=tmp_path / "runs",
        repo_root=repo_root,
        default_reference_fasta=default_reference,
        dataset_registry=BackgroundDatasetRegistry(repo_root / "data" / "background_sets"),
    )

    job = runner.create_job(
        {"target": b">target\nAAAA\n"},
        AnalysisOptions(background_dataset="demo", target_date="2024-01-02"),
    )

    assert (job.inputs_dir / "background.fasta").read_text() == ">bg\nAAAA\n"
    assert (job.inputs_dir / "tree_dates.csv").read_text() == (
        "name,date\nbg,1968\ntarget,2024-01-02\n"
    )
    input_manifest = json.loads((job.inputs_dir / "input_manifest.json").read_text())
    assert input_manifest["background_dataset"]["id"] == "demo"
    assert set(input_manifest["auto_prepared_inputs"]) == {
        "background",
        "tree_date_metadata",
    }


def test_uploaded_background_and_tree_dates_override_dataset_defaults(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "data" / "background_sets" / "demo" / "v1"
    dataset_root.mkdir(parents=True)
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "id": "demo",
                "label": "Demo",
                "version": "v1",
                "background_fasta": "background.fasta",
                "metadata_csv": "metadata.csv",
            }
        )
    )
    (dataset_root / "background.fasta").write_text(">dataset\nAAAA\n")
    (dataset_root / "metadata.csv").write_text("name,date\ndataset,1968\n")
    default_reference = tmp_path / "default_reference.fasta"
    default_reference.write_text(">default\nAAAA\n")
    runner = AnalysisRunner(
        runs_root=tmp_path / "runs",
        repo_root=repo_root,
        default_reference_fasta=default_reference,
        dataset_registry=BackgroundDatasetRegistry(repo_root / "data" / "background_sets"),
    )

    job = runner.create_job(
        {
            "target": b">target\nAAAA\n",
            "background": b">uploaded\nCCCC\n",
            "tree_date_metadata": b"name,date\nuploaded,2020\n",
        },
        AnalysisOptions(background_dataset="demo", target_date="2024-01-02"),
    )

    assert (job.inputs_dir / "background.fasta").read_text() == ">uploaded\nCCCC\n"
    assert (job.inputs_dir / "tree_dates.csv").read_text() == "name,date\nuploaded,2020\n"
    input_manifest = json.loads((job.inputs_dir / "input_manifest.json").read_text())
    assert input_manifest["auto_prepared_inputs"] == []


def test_parse_manifest_and_list_result_files(tmp_path):
    runner = AnalysisRunner(runs_root=tmp_path / "runs", repo_root=tmp_path)
    job = runner.create_job(
        {
            "target": b">target\nAAAA\n",
            "reference": b">reference\nAAAA\n",
        },
        AnalysisOptions(),
    )
    manifest = {"counts": {"targets": 1}, "outputs": {"report": "report.html"}}
    (job.results_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (job.results_dir / "report.html").write_text("<html></html>")

    assert runner.parse_manifest(job.run_id) == manifest
    files = runner.list_result_files(job.run_id)
    assert {item["name"] for item in files} == {"report.html", "run_manifest.json"}
    assert runner.result_file_path(job.run_id, "report.html") == job.results_dir / "report.html"
