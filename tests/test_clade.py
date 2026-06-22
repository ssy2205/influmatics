import subprocess

import pytest

from influmatics.clade import (
    NextcladeError,
    aa_mutations_to_rows,
    build_nextclade_command,
    clade_assignments_to_rows,
    ensure_nextclade_available,
    parse_nextclade_aa_mutations,
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


def test_build_nextclade_command_with_input_dataset():
    command = build_nextclade_command(
        "input.fasta", "out", input_dataset="/data/flu_h3n2_ha"
    )

    assert command == [
        "nextclade",
        "run",
        "--output-all",
        "out",
        "--input-dataset",
        "/data/flu_h3n2_ha",
        "input.fasta",
    ]


def test_build_nextclade_command_rejects_dataset_and_input_dataset():
    with pytest.raises(ValueError, match="only one"):
        build_nextclade_command(
            "input.fasta", "out", dataset="flu_h3n2_ha", input_dataset="/data/x"
        )


def test_ensure_nextclade_available_raises_when_missing(monkeypatch):
    monkeypatch.setattr("influmatics.clade.shutil.which", lambda _: None)

    with pytest.raises(NextcladeError, match="Nextclade was not found"):
        ensure_nextclade_available()


def test_run_nextclade_returns_metadata(tmp_path, monkeypatch):
    input_fasta = tmp_path / "input.fasta"
    input_fasta.write_text(">sample\nACGT\n")
    monkeypatch.setattr("influmatics.clade.shutil.which", lambda _: "/usr/bin/nextclade")

    def fake_run(command, check, stdout, stderr, text, timeout=None):
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

    def fake_run(command, check, stdout, stderr, text, timeout=None):
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


def test_parse_nextclade_aa_mutations_expands_substitutions_and_deletions(tmp_path):
    tsv = tmp_path / "nextclade.tsv"
    tsv.write_text(
        "seqName\tclade\taaSubstitutions\taaDeletions\n"
        "demoH3\t3C.2a\tHA1:N145K,NA:H275Y,M2:S31N\tHA1:K144-\n"
        "demoEmpty\t3C.2a\t\t\n"
    )

    mutations = parse_nextclade_aa_mutations(tsv)

    # demoEmpty contributes nothing; demoH3 yields 3 subs + 1 deletion.
    assert {(m.seq_id, m.gene, m.position, m.mutation, m.mutation_type) for m in mutations} == {
        ("demoH3", "HA1", 145, "N145K", "substitution"),
        ("demoH3", "NA", 275, "H275Y", "substitution"),
        ("demoH3", "M2", 31, "S31N", "substitution"),
        ("demoH3", "HA1", 144, "K144-", "deletion"),
    }
    assert all(m.coordinate_space == "aa" for m in mutations)


def test_parse_nextclade_aa_mutations_can_skip_deletions(tmp_path):
    tsv = tmp_path / "nextclade.tsv"
    tsv.write_text(
        "seqName\taaSubstitutions\taaDeletions\n"
        "demoH3\tHA1:N145K\tHA1:K144-\n"
    )

    mutations = parse_nextclade_aa_mutations(tsv, include_deletions=False)

    assert [m.mutation for m in mutations] == ["N145K"]


def test_parse_nextclade_aa_mutations_skips_malformed_tokens(tmp_path):
    tsv = tmp_path / "nextclade.tsv"
    # A token with no gene prefix and a stray empty fragment must be skipped
    # rather than crashing the parse.
    tsv.write_text(
        "seqName\taaSubstitutions\n"
        "demoH3\tN145K,,HA1:S193F\n"
    )

    mutations = parse_nextclade_aa_mutations(tsv)

    assert [m.mutation for m in mutations] == ["S193F"]


def test_parse_nextclade_aa_mutations_requires_aa_columns(tmp_path):
    tsv = tmp_path / "nextclade.tsv"
    tsv.write_text("seqName\tclade\ndemoH3\t3C.2a\n")

    with pytest.raises(NextcladeError, match="amino-acid change columns"):
        parse_nextclade_aa_mutations(tsv)


def test_aa_mutations_to_rows_is_scanner_ready(tmp_path):
    tsv = tmp_path / "nextclade.tsv"
    tsv.write_text("seqName\taaSubstitutions\ndemoH3\tHA1:N145K\n")

    rows = aa_mutations_to_rows(parse_nextclade_aa_mutations(tsv))

    assert rows == [
        {
            "seq_id": "demoH3",
            "gene": "HA1",
            "position": 145,
            "mutation": "N145K",
            "mutation_type": "substitution",
            "coordinate_space": "aa",
        }
    ]


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
