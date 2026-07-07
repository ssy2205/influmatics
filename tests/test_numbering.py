import pytest

from influmatics.numbering import (
    NumberingEntry,
    aligned_to_ungapped_positions,
    build_numbering_map,
    numbering_rows_to_tsv,
    read_numbering_table,
    ungapped_to_aligned_positions,
)


def test_coordinate_maps_handle_gaps():
    aligned = "A-CG-"

    assert ungapped_to_aligned_positions(aligned) == {1: 1, 2: 3, 3: 4}
    assert aligned_to_ungapped_positions(aligned) == {
        1: 1,
        2: None,
        3: 2,
        4: 3,
        5: None,
    }


def test_read_numbering_table(tmp_path):
    table = tmp_path / "numbering.tsv"
    table.write_text(
        "scheme\tgene\treference_position\tnumbering_label\tnote\n"
        "H3\tHA\t145\t145A\tantigenic site\n"
    )

    entries = read_numbering_table(table)

    assert entries == [NumberingEntry("H3", "HA", 145, "145A", "antigenic site")]


def test_read_numbering_table_requires_columns(tmp_path):
    table = tmp_path / "bad.tsv"
    table.write_text("scheme\tgene\nH3\tHA\n")

    with pytest.raises(ValueError, match="missing columns"):
        read_numbering_table(table)


def test_read_numbering_table_rejects_non_integer_position(tmp_path):
    table = tmp_path / "bad.tsv"
    table.write_text(
        "scheme\tgene\treference_position\tnumbering_label\n"
        "H3\tHA\tfoo\t1\n"
    )

    with pytest.raises(ValueError, match="row 2: reference_position must be an integer"):
        read_numbering_table(table)


def test_read_numbering_table_rejects_non_positive_position(tmp_path):
    table = tmp_path / "bad.tsv"
    table.write_text(
        "scheme\tgene\treference_position\tnumbering_label\n"
        "H3\tHA\t0\t0\n"
    )

    with pytest.raises(ValueError, match="must be 1-based and positive"):
        read_numbering_table(table)


def test_build_numbering_map_preserves_insertion_style_labels():
    # Real H3 HA numbering uses letter suffixes for inserted residues
    # (e.g. 133, 133a). The label is carried verbatim from the curated table
    # while the integer reference_position resolves the alignment column.
    entries = [
        NumberingEntry("H3", "HA", 3, "133"),
        NumberingEntry("H3", "HA", 4, "133a"),
    ]

    rows = build_numbering_map("ACGT", entries)

    assert [(r.numbering_label, r.alignment_position) for r in rows] == [
        ("133", 3),
        ("133a", 4),
    ]


def test_build_numbering_map_filters_and_maps_alignment_positions():
    entries = [
        NumberingEntry("H3", "HA", 1, "1"),
        NumberingEntry("H3", "HA", 2, "2"),
        NumberingEntry("H1", "HA", 2, "2"),
        NumberingEntry("H3", "NA", 2, "2"),
        NumberingEntry("H3", "HA", 99, "99"),
    ]

    rows = build_numbering_map("A-C", entries, scheme="H3", gene="HA")

    assert [row.numbering_label for row in rows] == ["1", "2", "99"]
    assert [row.alignment_position for row in rows] == [1, 3, None]
    assert [row.reference_base for row in rows] == ["A", "C", None]


def test_numbering_rows_to_tsv_formats_missing_values():
    rows = build_numbering_map("A-C", [NumberingEntry("H3", "HA", 99, "99")])

    assert numbering_rows_to_tsv(rows) == [
        {
            "scheme": "H3",
            "gene": "HA",
            "reference_position": 99,
            "numbering_label": "99",
            "alignment_position": "",
            "reference_base": "",
            "note": "",
        }
    ]
