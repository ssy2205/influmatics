"""Antiviral resistance marker scanning."""

from __future__ import annotations


def marker_matches(mutation: str, markers: set[str]) -> bool:
    """Check whether a mutation string matches a curated marker."""

    return mutation.upper() in {marker.upper() for marker in markers}
