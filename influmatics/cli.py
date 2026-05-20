"""Command line interface for Influmatics."""

from __future__ import annotations

import argparse
from pathlib import Path

from .antigenic import (
    antigenic_hits_to_rows,
    read_antigenic_sites,
    read_mutation_rows as read_antigenic_mutation_rows,
    scan_antigenic_sites,
)
from .io import read_sequences, write_tsv
from .mutations import call_mutations_for_alignment, mutations_to_rows
from .qc import assess_sequences, qc_to_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="influmatics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    qc_parser = subparsers.add_parser("qc", help="Run basic sequence QC")
    qc_parser.add_argument("input", help="FASTA, FASTQ, CSV, or TSV input")
    qc_parser.add_argument("--out", required=True, help="Output QC summary TSV")
    qc_parser.add_argument("--min-length", type=int, default=500)
    qc_parser.add_argument("--max-ambiguous-fraction", type=float, default=0.05)

    antigenic_parser = subparsers.add_parser("antigenic", help="Scan antigenic-site mutations")
    antigenic_parser.add_argument("mutations", help="Mutation TSV")
    antigenic_parser.add_argument("--sites", required=True, help="Antigenic site JSON")
    antigenic_parser.add_argument("--out", required=True, help="Output antigenic hit TSV")

    mutation_parser = subparsers.add_parser("mutations", help="Call mutations from an aligned FASTA")
    mutation_parser.add_argument("alignment", help="Aligned FASTA")
    mutation_parser.add_argument("--reference-id", required=True)
    mutation_parser.add_argument("--out", required=True, help="Output mutation TSV")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "qc":
        records = read_sequences(args.input)
        results = assess_sequences(
            records,
            min_length=args.min_length,
            max_ambiguous_fraction=args.max_ambiguous_fraction,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_tsv(qc_to_rows(results), args.out)
        return 0

    if args.command == "antigenic":
        mutation_rows = read_antigenic_mutation_rows(args.mutations)
        definition = read_antigenic_sites(args.sites)
        hits = scan_antigenic_sites(mutation_rows, definition)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_tsv(antigenic_hits_to_rows(hits), args.out)
        return 0

    if args.command == "mutations":
        records = read_sequences(args.alignment)
        mutations = call_mutations_for_alignment(records, args.reference_id)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_tsv(mutations_to_rows(mutations), args.out)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
