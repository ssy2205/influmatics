"""Input readers and identifier normalization."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SeqRecord:
    """Small sequence record used before optional Biopython integration."""

    seq_id: str
    sequence: str
    description: str = ""

    @property
    def norm_id(self) -> str:
        return normalize_id(self.seq_id)


def normalize_id(seq_id: str) -> str:
    """Normalize sequence identifiers for table joins and reports."""

    cleaned = seq_id.strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9_.|:-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def read_sequences(path: str | Path) -> list[SeqRecord]:
    """Read FASTA, FASTQ, CSV, or TSV sequence inputs."""

    input_path = Path(path)
    suffixes = [suffix.lower() for suffix in input_path.suffixes]
    if not suffixes:
        raise ValueError(f"Cannot infer input format without file extension: {input_path}")

    if suffixes[-1] in {".fa", ".fasta", ".fna", ".fas"}:
        return list(read_fasta(input_path))
    if suffixes[-1] in {".fq", ".fastq"} or suffixes[-2:] in [[".fq", ".gz"], [".fastq", ".gz"]]:
        return list(read_fastq(input_path))
    if suffixes[-1] == ".csv":
        return list(read_delimited_sequences(input_path, delimiter=","))
    if suffixes[-1] == ".tsv":
        return list(read_delimited_sequences(input_path, delimiter="\t"))
    raise ValueError(f"Unsupported sequence input format: {input_path}")


def read_fasta(path: str | Path) -> Iterable[SeqRecord]:
    """Read FASTA records with a minimal parser."""

    current_header: str | None = None
    chunks: list[str] = []
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    yield _record_from_header(current_header, chunks)
                current_header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if current_header is not None:
        yield _record_from_header(current_header, chunks)


def read_fastq(path: str | Path) -> Iterable[SeqRecord]:
    """Read FASTQ records, including gzip-compressed files."""

    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        while True:
            header = handle.readline().rstrip()
            if not header:
                break
            sequence = handle.readline().rstrip()
            plus = handle.readline().rstrip()
            quality = handle.readline().rstrip()
            if not header.startswith("@") or not plus.startswith("+") or len(sequence) != len(quality):
                raise ValueError(f"Invalid FASTQ record near header: {header}")
            yield _record_from_header(header[1:], [sequence])


def read_delimited_sequences(path: str | Path, delimiter: str) -> Iterable[SeqRecord]:
    """Read CSV/TSV with id and sequence columns or two-column rows."""

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames and {"id", "sequence"}.issubset(set(reader.fieldnames)):
            for row in reader:
                yield SeqRecord(row["id"], row["sequence"], row.get("description", ""))
            return

    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if len(row) < 2:
                continue
            yield SeqRecord(row[0], row[1])


def write_tsv(rows: Iterable[dict[str, object]], path: str | Path) -> None:
    """Write dictionaries to TSV, preserving the first row's column order."""

    rows = list(rows)
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _record_from_header(header: str, chunks: list[str]) -> SeqRecord:
    parts = header.split(maxsplit=1)
    seq_id = parts[0]
    description = parts[1] if len(parts) > 1 else ""
    return SeqRecord(seq_id=seq_id, sequence="".join(chunks).upper(), description=description)
