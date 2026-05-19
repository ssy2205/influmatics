from influmatics.io import SeqRecord, normalize_id, read_fasta
from influmatics.mutations import call_nt_mutations
from influmatics.qc import assess_sequence


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
    assert result.pass_qc is False


def test_call_nt_mutations_uses_reference_coordinates():
    reference = SeqRecord("ref", "ACG-T")
    sample = SeqRecord("sample", "ATGGT")

    mutations = call_nt_mutations(reference, sample)

    assert [mutation.mutation for mutation in mutations] == ["C2T", "ins4G"]
