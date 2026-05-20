"""Report helpers."""

from __future__ import annotations

import csv
import html
from pathlib import Path


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    """Read a TSV file into rows."""

    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def render_table(rows: list[dict[str, str]], max_rows: int = 50) -> str:
    """Render rows as a compact HTML table."""

    if not rows:
        return "<p>No rows.</p>"
    columns = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows[:max_rows]:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    extra = ""
    if len(rows) > max_rows:
        extra = f"<p>Showing {max_rows} of {len(rows)} rows.</p>"
    return f"{extra}<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_section(title: str, rows: list[dict[str, str]], max_rows: int = 50) -> str:
    """Render one report section."""

    escaped_title = html.escape(title)
    return "\n".join(
        [
            f"<section><h2>{escaped_title}</h2>",
            f"<p>{len(rows)} rows</p>",
            render_table(rows, max_rows=max_rows),
            "</section>",
        ]
    )


def build_tsv_report(
    title: str,
    sections: list[tuple[str, list[dict[str, str]]]],
    max_rows: int = 50,
) -> str:
    """Build the body HTML for a multi-section TSV report."""

    rendered_sections = [render_section(name, rows, max_rows=max_rows) for name, rows in sections]
    return "\n".join([f"<h1>{html.escape(title)}</h1>", *rendered_sections])


def write_basic_html(title: str, body_html: str, output_path: str | Path) -> Path:
    """Write a small standalone HTML report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html lang=\"en\">",
                "<head>",
                "<meta charset=\"utf-8\">",
                f"<title>{html.escape(title)}</title>",
                "<style>",
                "body{font-family:Arial,sans-serif;margin:2rem;line-height:1.4}",
                "table{border-collapse:collapse;width:100%;margin:1rem 0}",
                "th,td{border:1px solid #ddd;padding:.4rem;text-align:left}",
                "th{background:#f6f6f6}",
                "section{margin:2rem 0}",
                "</style>",
                "</head>",
                "<body>",
                body_html,
                "</body>",
                "</html>",
            ]
        )
    )
    return path
