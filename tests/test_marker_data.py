"""Regression guards for the shipped marker/site data files.

These lock the curated tables in ``data/markers/`` so they can never
silently regress to the empty placeholders the repo started with. They
load the files through the real reader functions (not raw json/csv) so a
schema break is caught here too.
"""

from pathlib import Path

import pytest

from influmatics.antigenic import read_antigenic_sites, scan_antigenic_sites
from influmatics.resistance import read_antiviral_markers, scan_resistance_markers

MARKERS_DIR = Path(__file__).resolve().parents[1] / "data" / "markers"


def test_h3n2_antigenic_sites_are_curated():
    definition = read_antigenic_sites(MARKERS_DIR / "antigenic_sites_h3n2.json")

    assert definition.numbering == "H3"
    # The seven key cluster-transition residues from the curation report.
    assert definition.sites["A"] == [145]
    assert definition.sites["B"] == [155, 156, 158, 159, 189, 193]
    # Guard against a regression to the all-empty placeholder.
    assert any(definition.sites.values())


def test_h1n1_antigenic_sites_are_curated():
    definition = read_antigenic_sites(MARKERS_DIR / "antigenic_sites_h1n1.json")

    assert definition.numbering == "H1"
    assert set(definition.sites) == {"Sa", "Sb", "Ca1", "Ca2", "Cb"}
    assert all(positions for positions in definition.sites.values())
    # Spot-check a representative residue in each named site.
    assert 128 in definition.sites["Sa"]
    assert 198 in definition.sites["Sb"]
    assert 122 in definition.sites["Cb"]


def test_antiviral_markers_are_curated():
    markers = read_antiviral_markers(MARKERS_DIR / "antiviral_markers.tsv")

    labels = {m.mutation for m in markers}
    # Canonical resistance markers that must be present.
    assert {"H275Y", "S31N"} <= labels
    classes = {m.drug_class for m in markers}
    assert {"neuraminidase_inhibitor", "adamantane"} <= classes
    # Host-adaptation markers are labelled distinctly so they are not
    # mistaken for direct antiviral resistance.
    assert any(m.drug_class == "host_adaptation" for m in markers)


def test_curated_sites_and_markers_produce_hits():
    """End-to-end: curated tables must actually match real mutation labels."""

    h3 = read_antigenic_sites(MARKERS_DIR / "antigenic_sites_h3n2.json")
    markers = read_antiviral_markers(MARKERS_DIR / "antiviral_markers.tsv")

    aa_rows = [
        {"seq_id": "s", "mutation": "K145N", "position": "145", "coordinate_space": "aa"},
        {"seq_id": "s", "mutation": "T100A", "position": "100", "coordinate_space": "aa"},
    ]
    antigenic_hits = scan_antigenic_sites(aa_rows, h3)
    assert [(h.mutation, h.site) for h in antigenic_hits] == [("K145N", "A")]

    resistance_hits = scan_resistance_markers(
        [
            {"seq_id": "s", "mutation": "H275Y", "coordinate_space": "aa"},
            {"seq_id": "s", "mutation": "S31N", "coordinate_space": "aa"},
            {"seq_id": "s", "mutation": "Z999Q", "coordinate_space": "aa"},
        ],
        markers,
    )
    assert {h.mutation for h in resistance_hits} == {"H275Y", "S31N"}


@pytest.mark.parametrize(
    "filename",
    ["antigenic_sites_h3n2.json", "antigenic_sites_h1n1.json", "antiviral_markers.tsv"],
)
def test_marker_files_exist(filename):
    assert (MARKERS_DIR / filename).is_file()
