import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "legacy" / "h3n2_ha_analysis.py"
SPEC = importlib.util.spec_from_file_location("h3n2_ha_analysis", MODULE_PATH)
h3n2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(h3n2)


def test_load_treetime_dates_accepts_numeric_and_calendar_columns(tmp_path):
    dates_path = tmp_path / "dates.tsv"
    dates_path.write_text(
        "#node\tdate\tnumeric date\n"
        "NODE_1\t2024-01-15\t2024.038251\n"
        "NODE_2\t2023-12-01\n",
        encoding="utf-8",
    )

    dates = h3n2.load_treetime_dates(dates_path)

    assert dates["NODE_1"] == 2024.038251
    assert 2023.91 < dates["NODE_2"] < 2023.93


def test_render_uses_treetime_comment_dates_when_date_table_is_empty(tmp_path):
    tree_path = tmp_path / "timetree.nexus"
    tree_path.write_text(
        "#NEXUS\n"
        "Begin trees;\n"
        "Tree tree1 = ((A:0.1[&date=2020.0],B:0.2[&date=2021.0])"
        "NODE_1:0.3[&date=2019.5],C:0.4[&date=2022.0],D:50.0[&date=2158.25])"
        "NODE_0:0.0[&date=2019.0];\n"
        "End;\n",
        encoding="utf-8",
    )
    out_png = tmp_path / "tree.png"

    stats = h3n2.render_newick_tree_png(
        tree_path,
        "nexus",
        {"A": "test", "B": "test", "C": "test", "D": "test"},
        out_png,
        x_by_name={},
        plot_style="dashboard",
    )

    assert out_png.is_file()
    assert stats["tree_calendar_coordinates"] is True
    assert stats["tree_x_min"] == 2019.0
    assert stats["tree_x_max"] == 2022.0
    assert stats["tree_x_span"] == 3.0


def test_write_tree_as_newick_sanitizes_treetime_node_labels(tmp_path):
    tree_path = tmp_path / "timetree.nexus"
    tree_path.write_text(
        "#NEXUS\n"
        "Begin trees;\n"
        "Tree tree1 = ((A:1[&date=2020],B:1[&date=2020])"
        "NODE_1:1[&date=2019],C:1[&date=2021])NODE_0:0[&date=2018];\n"
        "End;\n",
        encoding="utf-8",
    )
    out_newick = tmp_path / "tree.newick"

    h3n2.write_tree_as_newick(tree_path, "nexus", out_newick)

    text = out_newick.read_text(encoding="utf-8")
    assert out_newick.stat().st_size > 0
    assert "A:1.00000" in text
    assert "B:1.00000" in text
    assert "NODE_0" not in text
    parsed = h3n2.Phylo.read(str(out_newick), "newick")
    assert len(parsed.get_terminals()) == 3


def test_write_tree_as_newick_keeps_existing_file_when_export_fails(tmp_path, monkeypatch):
    tree_path = tmp_path / "timetree.nexus"
    tree_path.write_text(
        "#NEXUS\n"
        "Begin trees;\n"
        "Tree tree1 = (A:1[&date=2020],B:1[&date=2020])NODE_0:0[&date=2018];\n"
        "End;\n",
        encoding="utf-8",
    )
    out_newick = tmp_path / "tree.newick"
    out_newick.write_text("(Old:1);\n", encoding="utf-8")

    def fail_write(*_args, **_kwargs):
        raise RuntimeError("simulated writer failure")

    monkeypatch.setattr(h3n2.Phylo, "write", fail_write)

    with pytest.raises(RuntimeError):
        h3n2.write_tree_as_newick(tree_path, "nexus", out_newick)

    assert out_newick.read_text(encoding="utf-8") == "(Old:1);\n"
