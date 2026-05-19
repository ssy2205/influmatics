"""Report helpers."""

from __future__ import annotations

from pathlib import Path


def write_basic_html(title: str, body_html: str, output_path: str | Path) -> Path:
    """Write a small standalone HTML report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html lang=\"en\">",
                "<head><meta charset=\"utf-8\"><title>{}</title></head>".format(title),
                "<body>",
                body_html,
                "</body>",
                "</html>",
            ]
        )
    )
    return path
