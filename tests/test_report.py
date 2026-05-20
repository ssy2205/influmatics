from influmatics.report import build_tsv_report, read_tsv, render_table, write_basic_html


def test_read_tsv(tmp_path):
    path = tmp_path / "table.tsv"
    path.write_text("seq_id\tvalue\nsample\tgood\n")

    assert read_tsv(path) == [{"seq_id": "sample", "value": "good"}]


def test_render_table_escapes_values():
    html = render_table([{"name": "<sample>", "value": "A&B"}])

    assert "&lt;sample&gt;" in html
    assert "A&amp;B" in html


def test_render_table_limits_rows():
    rows = [{"id": str(index)} for index in range(3)]

    html = render_table(rows, max_rows=2)

    assert "Showing 2 of 3 rows" in html
    assert "<td>2</td>" not in html


def test_build_tsv_report_renders_sections():
    html = build_tsv_report("Report", [("QC", [{"seq_id": "sample"}])])

    assert "<h1>Report</h1>" in html
    assert "<h2>QC</h2>" in html
    assert "1 rows" in html


def test_write_basic_html(tmp_path):
    output = tmp_path / "report.html"

    path = write_basic_html("Title <x>", "<p>Body</p>", output)

    assert path == output
    text = output.read_text()
    assert "<title>Title &lt;x&gt;</title>" in text
    assert "<p>Body</p>" in text
