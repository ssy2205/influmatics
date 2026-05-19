"""Antigenic-site mutation scanning."""

from __future__ import annotations


def mutation_in_sites(position: int, sites: dict[str, list[int]]) -> list[str]:
    """Return antigenic site names that contain a position."""

    return [name for name, positions in sites.items() if position in positions]
