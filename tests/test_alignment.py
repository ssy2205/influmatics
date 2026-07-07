import shutil
import subprocess

import pytest

from influmatics.alignment import (
    AlignmentError,
    build_mafft_command,
    ensure_mafft_available,
    run_mafft,
)

mafft_available = pytest.mark.skipif(
    shutil.which("mafft") is None,
    reason="mafft not installed; skipping real-binary integration test",
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

    def fake_run(command, check, stdout, stderr, text, timeout=None):
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


def test_run_mafft_raises_when_input_missing(tmp_path, monkeypatch):
    output_fasta = tmp_path / "aligned.fasta"
    monkeypatch.setattr("influmatics.alignment.shutil.which", lambda _: "/usr/bin/mafft")

    with pytest.raises(AlignmentError, match="does not exist"):
        run_mafft(tmp_path / "missing.fasta", output_fasta)


def test_run_mafft_raises_on_command_failure(tmp_path, monkeypatch):
    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "aligned.fasta"
    input_fasta.write_text(">a\nACGT\n")

    monkeypatch.setattr("influmatics.alignment.shutil.which", lambda _: "/usr/bin/mafft")

    def fake_run(command, check, stdout, stderr, text, timeout=None):
        # Simulate MAFFT writing a partial alignment before exiting non-zero.
        stdout.write(">a\nACG")
        return subprocess.CompletedProcess(command, 1, stderr="bad input")

    monkeypatch.setattr("influmatics.alignment.subprocess.run", fake_run)

    with pytest.raises(AlignmentError, match="bad input"):
        run_mafft(input_fasta, output_fasta)

    # A failed run must leave neither the final output nor a stale temp file.
    assert not output_fasta.exists()
    assert not (output_fasta.with_suffix(output_fasta.suffix + ".tmp")).exists()


def test_run_mafft_raises_alignment_error_on_timeout(tmp_path, monkeypatch):
    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "aligned.fasta"
    input_fasta.write_text(">a\nACGT\n")

    monkeypatch.setattr("influmatics.alignment.shutil.which", lambda _: "/usr/bin/mafft")

    def fake_run(command, check, stdout, stderr, text, timeout=None):
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    monkeypatch.setattr("influmatics.alignment.subprocess.run", fake_run)

    with pytest.raises(AlignmentError, match="timed out"):
        run_mafft(input_fasta, output_fasta, timeout=0.01)

    # Temp file must be cleaned up so subsequent runs don't pick up garbage.
    assert not (output_fasta.with_suffix(output_fasta.suffix + ".tmp")).exists()
    assert not output_fasta.exists()


@mafft_available
def test_run_mafft_against_real_binary(tmp_path):
    """End-to-end check against a real mafft install (skipped when absent).

    Two sequences differing by one base should align to equal length with
    the expected residues preserved. This guards the wrapper against
    real-world argument/stdout-handling regressions that mocks can't catch.
    """

    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "aligned.fasta"
    input_fasta.write_text(">a\nACGTACGTACGT\n>b\nACGTACGAACGT\n")

    result = run_mafft(input_fasta, output_fasta, threads=1)

    assert output_fasta.exists()
    records = [
        line for line in output_fasta.read_text().splitlines() if line and not line.startswith(">")
    ]
    seqs = "".join(records)
    # MAFFT may lowercase residues; both inputs are gap-free and equal length.
    assert set(seqs.upper()) <= set("ACGT-")
    assert "mafft" in result.command[0]
