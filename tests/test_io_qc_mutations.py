from influmatics.io import SeqRecord, normalize_id, read_fasta
from influmatics.mutations import call_nt_mutations
from influmatics.qc import assess_sequence, qc_to_rows
from influmatics.validation import require_valid_sequence_input, validate_sequence_input


def test_normalize_id_removes_spaces_and_symbols():
    assert normalize_id("sample 1 / H3N2") == "sample_1_H3N2"


def test_read_fasta(tmp_path):
    fasta = tmp_path / "example.fasta"
    fasta.write_text(">ref description\nACGT\n>sample\nACGA\n")

    records = list(read_fasta(fasta))

    assert [record.seq_id for record in records] == ["ref", "sample"]
    assert records[0].description == "description"
    assert records[1].sequence == "ACGA"


def test_assess_sequence_flags_invalid_characters():
    result = assess_sequence(SeqRecord("sample", "ACGTZ"), min_length=1)

    assert result.invalid_characters == "Z"
    assert result.invalid_count == 1
    assert result.fail_reasons == "invalid_characters"
    assert result.pass_qc is False


def test_assess_sequence_reports_fractions():
    result = assess_sequence(
        SeqRecord("sample", "ACGTNN--"),
        min_length=1,
        max_ambiguous_fraction=0.3,
    )

    assert result.ambiguous_fraction == 0.25
    assert result.gap_fraction == 0.25
    assert result.gc_fraction == 0.25
    assert "too_many_gaps" in result.fail_reasons


def test_qc_to_rows_formats_fraction_columns():
    result = assess_sequence(SeqRecord("sample", "ACGT"), min_length=1)

    rows = qc_to_rows([result])

    assert rows[0]["ambiguous_fraction"] == "0.000000"
    assert rows[0]["gc_fraction"] == "0.500000"


def test_validate_sequence_input_rejects_duplicate_normalized_ids(tmp_path):
    fasta = tmp_path / "duplicate.fasta"
    fasta.write_text(">sample_1\nACGT\n>sample/1\nACGT\n")

    result = validate_sequence_input(fasta)

    assert result.ok is False
    assert result.errors == ("Duplicate normalized IDs: sample_1",)


def test_require_valid_sequence_input_returns_records(tmp_path):
    fasta = tmp_path / "valid.fasta"
    fasta.write_text(">sample\nACGT\n")

    records = require_valid_sequence_input(fasta)

    assert records == [SeqRecord("sample", "ACGT")]


def test_call_nt_mutations_uses_reference_coordinates():
    reference = SeqRecord("ref", "ACG-T")
    sample = SeqRecord("sample", "ATGGT")

    mutations = call_nt_mutations(reference, sample)

    assert [mutation.mutation for mutation in mutations] == ["C2T", "ins4G"]
