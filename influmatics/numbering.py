"""Coordinate and numbering helpers."""

from __future__ import annotations


def ungapped_to_aligned_positions(aligned_sequence: str) -> dict[int, int]:
    """Map 1-based ungapped sequence positions to 1-based aligned coordinates."""

    mapping: dict[int, int] = {}
    ungapped_position = 0
    for aligned_position, base in enumerate(aligned_sequence, start=1):
        if base == "-":
            continue
        ungapped_position += 1
        mapping[ungapped_position] = aligned_position
    return mapping


def aligned_to_ungapped_positions(aligned_sequence: str) -> dict[int, int | None]:
    """Map 1-based aligned coordinates to 1-based ungapped positions."""

    mapping: dict[int, int | None] = {}
    ungapped_position = 0
    for aligned_position, base in enumerate(aligned_sequence, start=1):
        if base == "-":
            mapping[aligned_position] = None
            continue
        ungapped_position += 1
        mapping[aligned_position] = ungapped_position
    return mapping
