import sys

from influmatics.io import SeqRecord
from influmatics.web import (
    available_modules,
    bundled_marker_files,
    rows_to_tsv_text,
    run_qc_rows,
    scan_aa_table,
    selected_module_labels,
    streamlit_command,
)


def test_available_modules_contains_mvp_steps():
    keys = [module.key for module in available_modules()]

    assert keys == [
        "qc",
        "alignment",
        "mutations",
        "clade",
        "antigenic",
        "resistance",
        "report",
    ]


def test_selected_module_labels_ignores_unknown_keys():
    labels = selected_module_labels(["qc", "unknown", "report"])

    assert labels == ["QC", "Report"]


def test_run_qc_rows_flags_short_sequences():
    records = [SeqRecord("a", "ACGT" * 200), SeqRecord("b", "ACGT")]

    rows = run_qc_rows(records, min_length=500)

    by_id = {row["seq_id"]: row for row in rows}
    assert by_id["a"]["pass_qc"] is True
    assert by_id["b"]["pass_qc"] is False
    assert "too_short" in by_id["b"]["fail_reasons"]


def test_bundled_marker_files_point_at_shipped_data():
    files = bundled_marker_files()

    kinds = {f.kind for f in files}
    assert kinds == {"antigenic", "resistance"}
    # The curated tables must actually exist on disk.
    assert all(f.path.is_file() for f in files)


def test_scan_aa_table_dispatches_antigenic_and_resistance(tmp_path):
    aa_tsv = tmp_path / "aa.tsv"
    aa_tsv.write_text(
        "seq_id\tmutation\tposition\tcoordinate_space\n"
        "s\tN145K\t145\taa\n"
        "s\tH275Y\t275\taa\n"
        "s\tS31N\t31\taa\n"
    )
    files = {f.key: f for f in bundled_marker_files()}

    antigenic_hits = scan_aa_table(aa_tsv, files["h3n2_sites"])
    assert any(hit["mutation"] == "N145K" and hit["site"] == "A" for hit in antigenic_hits)

    resistance_hits = scan_aa_table(aa_tsv, files["antiviral"])
    assert {hit["mutation"] for hit in resistance_hits} == {"H275Y", "S31N"}


def test_rows_to_tsv_text_roundtrips():
    rows = [{"seq_id": "s", "mutation": "N145K"}]

    text = rows_to_tsv_text(rows)

    assert text.splitlines()[0] == "seq_id\tmutation"
    assert "s\tN145K" in text
    assert rows_to_tsv_text([]) == ""


def test_streamlit_command_invokes_module_runner():
    command = streamlit_command(port=8600, headless=True)

    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert command[4].endswith("web.py")
    assert "--server.port" in command and "8600" in command
    assert command[command.index("--server.headless") + 1] == "true"


def test_streamlit_command_headless_defaults_false():
    command = streamlit_command()

    assert command[command.index("--server.headless") + 1] == "false"
    assert "8501" in command
