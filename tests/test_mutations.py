import pytest

from influmatics.io import SeqRecord
from influmatics.mutations import (
    call_mutations_for_alignment,
    call_nt_mutations,
    mutations_to_rows,
)


def test_call_nt_mutations_reports_coordinates_and_types():
    reference = SeqRecord("ref", "ACG-T")
    sample = SeqRecord("sample", "ATGGT")

    mutations = call_nt_mutations(reference, sample)

    assert [mutation.mutation for mutation in mutations] == ["C2T", "ins4G"]
    assert [mutation.mutation_type for mutation in mutations] == ["substitution", "insertion"]
    assert [mutation.alignment_position for mutation in mutations] == [2, 4]
    assert [mutation.reference_position for mutation in mutations] == [2, None]
    assert [mutation.query_position for mutation in mutations] == [2, 4]


def test_call_nt_mutations_reports_deletion_query_coordinate_as_empty():
    reference = SeqRecord("ref", "ACGT")
    sample = SeqRecord("sample", "AC-T")

    mutations = call_nt_mutations(reference, sample)

    assert len(mutations) == 1
    assert mutations[0].mutation == "G3del"
    assert mutations[0].mutation_type == "deletion"
    assert mutations[0].reference_position == 3
    assert mutations[0].query_position is None


def test_call_mutations_for_alignment_rejects_non_unique_reference():
    records = [
        SeqRecord("ref", "ACGT"),
        SeqRecord("ref", "ACGT"),
        SeqRecord("sample", "ACGA"),
    ]

    with pytest.raises(ValueError, match="not unique"):
        call_mutations_for_alignment(records, "ref")


def test_call_mutations_for_alignment_rejects_unequal_alignment_lengths():
    records = [
        SeqRecord("ref", "ACGT"),
        SeqRecord("sample", "ACG"),
    ]

    with pytest.raises(ValueError, match="equal length"):
        call_mutations_for_alignment(records, "ref")


def test_mutations_to_rows_includes_coordinate_columns():
    mutation = call_nt_mutations(SeqRecord("ref", "ACGT"), SeqRecord("sample", "AC-T"))[0]

    rows = mutations_to_rows([mutation])

    assert rows == [
        {
            "seq_id": "sample",
            "mutation_type": "deletion",
            "position": 3,
            "alignment_position": 3,
            "reference_position": 3,
            "query_position": "",
            "reference": "G",
            "observed": "-",
            "mutation": "G3del",
            "coordinate_space": "nt",
        }
    ]
