"""Subtype/source screening placeholders."""

from __future__ import annotations


def parse_subtype_from_text(text: str) -> str | None:
    """Extract an influenza subtype such as H3N2 from free text."""

    import re

    match = re.search(r"\bH\d{1,2}N\d{1,2}\b", text, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None
