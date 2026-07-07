from pathlib import Path

from influmatics.antigenic import read_antigenic_sites
from influmatics.io import read_sequences
from influmatics.resistance import read_antiviral_markers


def test_h3n2_antigenic_sites_include_key_positions():
    definition = read_antigenic_sites("data/markers/antigenic_sites_h3n2.json")

    assert definition.numbering == "H3"
    assert 145 in definition.sites["A"]
    for position in [155, 156, 158, 159, 189, 193]:
        assert position in definition.sites["B"]


def test_h1n1_antigenic_sites_include_named_regions():
    definition = read_antigenic_sites("data/markers/antigenic_sites_h1n1.json")

    assert definition.numbering == "H1"
    assert {"Sa", "Sb", "Ca1", "Ca2", "Cb"}.issubset(definition.sites)
    assert 128 in definition.sites["Sa"]
    assert 198 in definition.sites["Sb"]
    assert 122 in definition.sites["Cb"]


def test_antiviral_marker_table_includes_nai_and_m2_markers():
    markers = read_antiviral_markers("data/markers/antiviral_markers.tsv")
    marker_keys = {(marker.gene, marker.mutation) for marker in markers}

    assert ("NA", "H275Y") in marker_keys
    assert ("M2", "S31N") in marker_keys


def test_public_reference_fastas_are_readable():
    reference_dir = Path("data/references")
    expected = [
        "FJ966974.1_H1N1pdm09_HA.fasta",
        "A_Aichi_1968_H3N2_HA.fasta",
    ]

    for filename in expected:
        records = read_sequences(reference_dir / filename)
        assert len(records) == 1
        assert len(records[0].sequence) > 1000
