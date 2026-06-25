"""Create a protein manifest TSV from FASTA input."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from influenza_protein_dl.sequence_features import feature_row


def read_fasta(path: str | Path) -> list[tuple[str, str, str]]:
    """Read FASTA records as (id, sequence, description)."""

    records: list[tuple[str, str, str]] = []
    current_header: str | None = None
    chunks: list[str] = []
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records.append(_record_from_header(current_header, chunks))
                current_header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if current_header is not None:
        records.append(_record_from_header(current_header, chunks))
    return records


def _record_from_header(header: str, chunks: list[str]) -> tuple[str, str, str]:
    parts = header.split(maxsplit=1)
    seq_id = parts[0]
    description = parts[1] if len(parts) > 1 else ""
    return seq_id, "".join(chunks), description


def write_tsv(rows: list[dict[str, object]], path: str | Path) -> None:
    """Write rows to TSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("")
        return
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Protein FASTA input")
    parser.add_argument("--out", required=True, help="Output manifest TSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = read_fasta(args.input)
    rows = [
        feature_row(seq_id=seq_id, sequence=sequence, description=description)
        for seq_id, sequence, description in records
    ]
    write_tsv(rows, args.out)
    print(f"Wrote {len(rows)} protein records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
