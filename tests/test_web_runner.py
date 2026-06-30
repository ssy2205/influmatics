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
            "tree_date_metadata": b"name,date\ntarget,2023\nbackground1,2022\nbackground2,2021\n",
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


def test_build_command_uses_env_nextclade_dataset(tmp_path, monkeypatch):
    script = tmp_path / "legacy" / "h3n2_ha_analysis.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n")
    default_reference = tmp_path / "default_reference.fasta"
    default_reference.write_text(">default\nAAAA\n")
    dataset_path = tmp_path / "nextclade" / "flu_h3n2_ha"
    dataset_path.mkdir(parents=True)
    monkeypatch.setenv("INFLUMATICS_NEXTCLADE_DATASET", str(dataset_path))
    runner = AnalysisRunner(
        runs_root=tmp_path / "runs",
        repo_root=tmp_path,
        legacy_script=script,
        default_reference_fasta=default_reference,
    )
    job = runner.create_job(
        {"target": b">target\nAAAA\n"},
        AnalysisOptions(clade_method="auto"),
    )

    cmd = runner.build_command(job)

    assert "--nextclade-dataset" in cmd
    assert str(dataset_path) in cmd


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


def test_background_registry_reads_external_dataset_root(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_dataset_root = repo_root / "data" / "background_sets" / "demo" / "v1"
    external_root = tmp_path / "volume" / "background_sets"
    external_dataset_root = external_root / "external" / "v1"
    repo_dataset_root.mkdir(parents=True)
    external_dataset_root.mkdir(parents=True)
    for dataset_root, dataset_id in [
        (repo_dataset_root, "demo"),
        (external_dataset_root, "external"),
    ]:
        (dataset_root / "manifest.json").write_text(
            json.dumps(
                {
                    "id": dataset_id,
                    "label": dataset_id.title(),
                    "version": "v1",
                    "background_fasta": "background.fasta",
                }
            )
        )
        (dataset_root / "background.fasta").write_text(f">{dataset_id}\nAAAA\n")
    monkeypatch.setenv("INFLUMATICS_BACKGROUND_SETS_ROOT", str(external_root))

    registry = BackgroundDatasetRegistry.for_repo(repo_root)

    assert {dataset.id for dataset in registry.list()} == {"demo", "external"}


def test_background_registry_defaults_to_largest_analysis_ready_dataset(tmp_path):
    root = tmp_path / "background_sets"
    for dataset_id, sequence_count in [
        ("h3n2_ha_demo_reference", 1),
        ("larger_public_background", 50),
        ("medium_public_background", 12),
    ]:
        dataset_root = root / dataset_id / "v1"
        dataset_root.mkdir(parents=True)
        (dataset_root / "manifest.json").write_text(
            json.dumps(
                {
                    "id": dataset_id,
                    "label": dataset_id,
                    "version": "v1",
                    "background_fasta": "background.fasta",
                    "sequence_count": sequence_count,
                }
            )
        )
        (dataset_root / "background.fasta").write_text(f">{dataset_id}\nAAAA\n")

    registry = BackgroundDatasetRegistry(root)

    assert registry.default_dataset_id() == "larger_public_background"


def test_treetime_requires_at_least_three_dated_tips(tmp_path):
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

    try:
        runner.create_job(
            {"target": b">target\nAAAA\n"},
            AnalysisOptions(
                background_dataset="demo",
                target_date="2024-01-02",
                tree_method="iqtree-treetime",
            ),
        )
    except ValueError as exc:
        assert "at least 3 dated tips" in str(exc)
    else:
        raise AssertionError("TreeTime run with fewer than 3 dated tips should fail")


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
