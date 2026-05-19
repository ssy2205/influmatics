"""Metadata summaries for phylogeography-style reports."""

from __future__ import annotations

from collections import Counter


def count_metadata_values(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    """Count non-empty metadata values in one field."""

    return dict(Counter(row[field] for row in rows if row.get(field)))
