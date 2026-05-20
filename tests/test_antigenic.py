import pytest

from influmatics.antigenic import (
    AntigenicSiteDefinition,
    antigenic_hits_to_rows,
    mutation_in_sites,
    read_antigenic_sites,
    read_mutation_rows,
    scan_antigenic_sites,
)


def test_mutation_in_sites_returns_all_matching_sites():
    sites = {"A": [1, 2], "B": [2, 3], "C": []}

    assert mutation_in_sites(2, sites) == ["A", "B"]


def test_read_antigenic_sites(tmp_path):
    site_json = tmp_path / "sites.json"
    site_json.write_text(
        '{"numbering": "H3", "note": "verify", "sites": {"A": [145], "B": ["155"]}}'
    )

    definition = read_antigenic_sites(site_json)

    assert definition == AntigenicSiteDefinition("H3", {"A": [145], "B": [155]}, "verify")


def test_read_antigenic_sites_requires_sites_object(tmp_path):
    site_json = tmp_path / "bad.json"
    site_json.write_text('{"numbering": "H3"}')

    with pytest.raises(ValueError, match="sites object"):
        read_antigenic_sites(site_json)


def test_read_mutation_rows_requires_columns(tmp_path):
    mutation_tsv = tmp_path / "mutations.tsv"
    mutation_tsv.write_text("seq_id\tmutation\nsample\tA145T\n")

    with pytest.raises(ValueError, match="missing columns"):
        read_mutation_rows(mutation_tsv)


def test_scan_antigenic_sites_matches_positions():
    mutation_rows = [
        {"seq_id": "sample1", "mutation": "A145T", "position": "145"},
        {"seq_id": "sample2", "mutation": "G200A", "position": "200"},
    ]
    definition = AntigenicSiteDefinition("H3", {"A": [145]}, "placeholder")

    hits = scan_antigenic_sites(mutation_rows, definition)

    assert len(hits) == 1
    assert hits[0].seq_id == "sample1"
    assert hits[0].site == "A"


def test_antigenic_hits_to_rows():
    hits = scan_antigenic_sites(
        [{"seq_id": "sample", "mutation": "A145T", "position": "145"}],
        AntigenicSiteDefinition("H3", {"A": [145]}, "placeholder"),
    )

    assert antigenic_hits_to_rows(hits) == [
        {
            "seq_id": "sample",
            "mutation": "A145T",
            "position": 145,
            "site": "A",
            "numbering": "H3",
            "note": "placeholder",
        }
    ]
