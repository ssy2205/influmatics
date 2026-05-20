import subprocess

import pytest

from influmatics.alignment import (
    AlignmentError,
    build_mafft_command,
    ensure_mafft_available,
    run_mafft,
)


def test_build_mafft_command_defaults_to_auto():
    command = build_mafft_command("input.fasta", threads=4)

    assert command == ["mafft", "--thread", "4", "--auto", "input.fasta"]


def test_build_mafft_command_can_reorder_without_auto():
    command = build_mafft_command("input.fasta", threads=2, auto=False, reorder=True)

    assert command == ["mafft", "--thread", "2", "--reorder", "input.fasta"]


def test_build_mafft_command_rejects_invalid_threads():
    with pytest.raises(ValueError, match="threads"):
        build_mafft_command("input.fasta", threads=0)


def test_ensure_mafft_available_raises_when_missing(monkeypatch):
    monkeypatch.setattr("influmatics.alignment.shutil.which", lambda _: None)

    with pytest.raises(AlignmentError, match="MAFFT was not found"):
        ensure_mafft_available()


def test_run_mafft_writes_output_and_returns_metadata(tmp_path, monkeypatch):
    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "nested" / "aligned.fasta"
    input_fasta.write_text(">a\nACGT\n>b\nACGA\n")

    monkeypatch.setattr("influmatics.alignment.shutil.which", lambda _: "/usr/bin/mafft")

    def fake_run(command, check, stdout, stderr, text):
        stdout.write(">a\nACGT\n>b\nACGA\n")
        return subprocess.CompletedProcess(command, 0, stderr="progress\n")

    monkeypatch.setattr("influmatics.alignment.subprocess.run", fake_run)

    result = run_mafft(input_fasta, output_fasta, threads=3, reorder=True)

    assert output_fasta.read_text() == ">a\nACGT\n>b\nACGA\n"
    assert result.output_fasta == output_fasta
    assert result.command == (
        "mafft",
        "--thread",
        "3",
        "--auto",
        "--reorder",
        str(input_fasta),
    )
    assert result.stderr == "progress\n"


def test_run_mafft_raises_on_command_failure(tmp_path, monkeypatch):
    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "aligned.fasta"
    input_fasta.write_text(">a\nACGT\n")

    monkeypatch.setattr("influmatics.alignment.shutil.which", lambda _: "/usr/bin/mafft")

    def fake_run(command, check, stdout, stderr, text):
        return subprocess.CompletedProcess(command, 1, stderr="bad input")

    monkeypatch.setattr("influmatics.alignment.subprocess.run", fake_run)

    with pytest.raises(AlignmentError, match="bad input"):
        run_mafft(input_fasta, output_fasta)
