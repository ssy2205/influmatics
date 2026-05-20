import subprocess

import pytest

from influmatics.clade import (
    NextcladeError,
    build_nextclade_command,
    clade_assignments_to_rows,
    ensure_nextclade_available,
    parse_nextclade_tsv,
    run_nextclade,
)


def test_build_nextclade_command_with_dataset():
    command = build_nextclade_command("input.fasta", "out", dataset="flu_h3n2_ha")

    assert command == [
        "nextclade",
        "run",
        "--output-all",
        "out",
        "--dataset-name",
        "flu_h3n2_ha",
        "input.fasta",
    ]


def test_ensure_nextclade_available_raises_when_missing(monkeypatch):
    monkeypatch.setattr("influmatics.clade.shutil.which", lambda _: None)

    with pytest.raises(NextcladeError, match="Nextclade was not found"):
        ensure_nextclade_available()


def test_run_nextclade_returns_metadata(tmp_path, monkeypatch):
    input_fasta = tmp_path / "input.fasta"
    input_fasta.write_text(">sample\nACGT\n")
    monkeypatch.setattr("influmatics.clade.shutil.which", lambda _: "/usr/bin/nextclade")

    def fake_run(command, check, stdout, stderr, text):
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="progress")

    monkeypatch.setattr("influmatics.clade.subprocess.run", fake_run)

    result = run_nextclade(input_fasta, tmp_path / "out", dataset="flu")

    assert result.output_dir == tmp_path / "out"
    assert result.stdout == "ok"
    assert "--dataset-name" in result.command


def test_run_nextclade_raises_on_failure(tmp_path, monkeypatch):
    input_fasta = tmp_path / "input.fasta"
    input_fasta.write_text(">sample\nACGT\n")
    monkeypatch.setattr("influmatics.clade.shutil.which", lambda _: "/usr/bin/nextclade")

    def fake_run(command, check, stdout, stderr, text):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad dataset")

    monkeypatch.setattr("influmatics.clade.subprocess.run", fake_run)

    with pytest.raises(NextcladeError, match="bad dataset"):
        run_nextclade(input_fasta, tmp_path / "out")


def test_parse_nextclade_tsv_extracts_core_fields(tmp_path):
    tsv = tmp_path / "nextclade.tsv"
    tsv.write_text("seqName\tclade\tqc.overallStatus\nsample\t3C.2a1b\tgood\n")

    assignments = parse_nextclade_tsv(tsv, dataset="flu_h3n2_ha")

    assert assignments[0].seq_id == "sample"
    assert assignments[0].clade == "3C.2a1b"
    assert assignments[0].qc_status == "good"
    assert assignments[0].dataset == "flu_h3n2_ha"


def test_parse_nextclade_tsv_requires_clade_column(tmp_path):
    tsv = tmp_path / "nextclade.tsv"
    tsv.write_text("seqName\tqc.overallStatus\nsample\tgood\n")

    with pytest.raises(NextcladeError, match="missing"):
        parse_nextclade_tsv(tsv)


def test_clade_assignments_to_rows(tmp_path):
    tsv = tmp_path / "nextclade.tsv"
    tsv.write_text("seqName\tclade\tqc.overallStatus\nsample\t3C.2a1b\tgood\n")

    rows = clade_assignments_to_rows(parse_nextclade_tsv(tsv, dataset="flu"))

    assert rows == [
        {
            "seq_id": "sample",
            "clade": "3C.2a1b",
            "qc_status": "good",
            "dataset": "flu",
        }
    ]
