import pytest

from influmatics.resistance import (
    ResistanceMarker,
    marker_matches,
    read_antiviral_markers,
    read_mutation_rows,
    resistance_hits_to_rows,
    scan_resistance_markers,
)


def test_marker_matches_is_case_insensitive():
    assert marker_matches("h275y", {"H275Y"})


def test_read_antiviral_markers(tmp_path):
    table = tmp_path / "markers.tsv"
    table.write_text(
        "drug_class\tgene\tsubtype\tnumbering\tmutation\tnote\n"
        "neuraminidase_inhibitor\tNA\tH1N1pdm\tNA\tH275Y\tverify numbering\n"
    )

    markers = read_antiviral_markers(table)

    assert markers == [
        ResistanceMarker(
            "neuraminidase_inhibitor",
            "NA",
            "H1N1pdm",
            "NA",
            "H275Y",
            "verify numbering",
        )
    ]


def test_read_antiviral_markers_requires_columns(tmp_path):
    table = tmp_path / "bad.tsv"
    table.write_text("mutation\nH275Y\n")

    with pytest.raises(ValueError, match="missing columns"):
        read_antiviral_markers(table)


def test_read_mutation_rows_requires_seq_id_and_mutation(tmp_path):
    table = tmp_path / "mutations.tsv"
    table.write_text("seq_id\tposition\nsample\t275\n")

    with pytest.raises(ValueError, match="seq_id and mutation"):
        read_mutation_rows(table)


def test_scan_resistance_markers_filters_gene_and_subtype():
    mutation_rows = [
        {"seq_id": "sample1", "mutation": "H275Y"},
        {"seq_id": "sample2", "mutation": "S31N"},
    ]
    markers = [
        ResistanceMarker("neuraminidase_inhibitor", "NA", "H1N1pdm", "NA", "H275Y"),
        ResistanceMarker("adamantane", "M2", "any", "M2", "S31N"),
    ]

    hits = scan_resistance_markers(mutation_rows, markers, gene="NA", subtype="H1N1pdm")

    assert len(hits) == 1
    assert hits[0].seq_id == "sample1"
    assert hits[0].mutation == "H275Y"


def test_resistance_hits_to_rows():
    mutation_rows = [{"seq_id": "sample", "mutation": "S31N"}]
    markers = [ResistanceMarker("adamantane", "M2", "any", "M2", "S31N", "common marker")]

    rows = resistance_hits_to_rows(scan_resistance_markers(mutation_rows, markers))

    assert rows == [
        {
            "seq_id": "sample",
            "mutation": "S31N",
            "drug_class": "adamantane",
            "gene": "M2",
            "subtype": "any",
            "numbering": "M2",
            "note": "common marker",
        }
    ]
