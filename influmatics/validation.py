"""Input validation before analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .io import SeqRecord, normalize_id, read_sequences


SUPPORTED_SUFFIXES = {
    ".fa",
    ".fasta",
    ".fna",
    ".fas",
    ".fq",
    ".fastq",
    ".csv",
    ".tsv",
}


@dataclass(frozen=True)
class InputValidation:
    path: Path
    # Immutable tuple so a frozen dataclass cannot be silently mutated by callers.
    records: tuple[SeqRecord, ...] = field(repr=False)
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


class InputValidationError(ValueError):
    """Raised when an input file cannot safely enter the analysis pipeline."""


def _empty(path: Path, message: str) -> InputValidation:
    return InputValidation(path, tuple(), (message,), ())


def validate_sequence_input(
    path: str | Path, allow_duplicate_ids: bool = False
) -> InputValidation:
    """Validate and parse a sequence input file."""

    input_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []

    if not input_path.exists():
        return _empty(input_path, f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        return _empty(input_path, f"Input path is not a file: {input_path}")
    if input_path.stat().st_size == 0:
        return _empty(input_path, f"Input file is empty: {input_path}")

    suffixes = [suffix.lower() for suffix in input_path.suffixes]
    if suffixes[-1:] == [".gz"] and len(suffixes) > 1:
        terminal_suffix = suffixes[-2]
    else:
        terminal_suffix = suffixes[-1] if suffixes else ""
    if not terminal_suffix or terminal_suffix not in SUPPORTED_SUFFIXES:
        errors.append(f"Unsupported input extension: {''.join(suffixes)}")

    try:
        records: list[SeqRecord] = read_sequences(input_path)
    except Exception as exc:
        errors.append(f"Could not parse input file: {exc}")
        return InputValidation(input_path, tuple(), tuple(errors), tuple(warnings))

    if not records:
        errors.append("No sequence records were found.")

    empty_ids = [index + 1 for index, record in enumerate(records) if not record.seq_id.strip()]
    if empty_ids:
        errors.append(f"Records with empty IDs: {','.join(map(str, empty_ids))}")

    empty_sequences = [record.seq_id for record in records if not record.sequence.strip()]
    if empty_sequences:
        errors.append(f"Records with empty sequences: {','.join(empty_sequences)}")

    normalized_ids = [normalize_id(record.seq_id) for record in records]
    # O(N) duplicate detection: Counter once instead of .count() per element.
    id_counts = Counter(normalized_ids)
    duplicate_ids = sorted(seq_id for seq_id, count in id_counts.items() if count > 1)
    if duplicate_ids and allow_duplicate_ids:
        warnings.append(f"Duplicate normalized IDs allowed: {','.join(duplicate_ids)}")
    elif duplicate_ids:
        errors.append(f"Duplicate normalized IDs: {','.join(duplicate_ids)}")

    return InputValidation(input_path, tuple(records), tuple(errors), tuple(warnings))


def require_valid_sequence_input(
    path: str | Path,
    allow_duplicate_ids: bool = False,
) -> list[SeqRecord]:
    """Return parsed records or raise a compact validation error."""

    result = validate_sequence_input(path, allow_duplicate_ids=allow_duplicate_ids)
    if not result.ok:
        raise InputValidationError("; ".join(result.errors))
    return list(result.records)
