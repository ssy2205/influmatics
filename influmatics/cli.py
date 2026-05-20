"""Command line interface for Influmatics."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import write_tsv
from .mutations import call_mutations_for_alignment, mutations_to_rows
from .qc import assess_sequences, qc_to_rows
from .validation import InputValidationError, require_valid_sequence_input, validate_sequence_input


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="influmatics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    qc_parser = subparsers.add_parser("qc", help="Run basic sequence QC")
    qc_parser.add_argument("input", help="FASTA, FASTQ, CSV, or TSV input")
    qc_parser.add_argument("--out", required=True, help="Output QC summary TSV")
    qc_parser.add_argument("--min-length", type=int, default=500)
    qc_parser.add_argument("--max-ambiguous-fraction", type=float, default=0.05)
    qc_parser.add_argument("--max-gap-fraction", type=float, default=0.05)
    qc_parser.add_argument("--allow-duplicate-ids", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate sequence input without running QC",
    )
    validate_parser.add_argument("input", help="FASTA, FASTQ, CSV, or TSV input")
    validate_parser.add_argument("--allow-duplicate-ids", action="store_true")

    mutation_parser = subparsers.add_parser(
        "mutations",
        help="Call mutations from an aligned FASTA",
    )
    mutation_parser.add_argument("alignment", help="Aligned FASTA")
    mutation_parser.add_argument("--reference-id", required=True)
    mutation_parser.add_argument("--out", required=True, help="Output mutation TSV")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "qc":
        try:
            records = require_valid_sequence_input(
                args.input,
                allow_duplicate_ids=args.allow_duplicate_ids,
            )
            results = assess_sequences(
                records,
                min_length=args.min_length,
                max_ambiguous_fraction=args.max_ambiguous_fraction,
                max_gap_fraction=args.max_gap_fraction,
            )
        except InputValidationError as exc:
            parser.error(str(exc))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_tsv(qc_to_rows(results), args.out)
        return 0

    if args.command == "validate":
        result = validate_sequence_input(
            args.input,
            allow_duplicate_ids=args.allow_duplicate_ids,
        )
        for warning in result.warnings:
            print(f"[WARN] {warning}")
        if not result.ok:
            for error in result.errors:
                print(f"[ERR] {error}")
            return 1
        print(f"[OK] {len(result.records)} sequence records validated")
        return 0

    if args.command == "mutations":
        records = require_valid_sequence_input(args.alignment, allow_duplicate_ids=False)
        mutations = call_mutations_for_alignment(records, args.reference_id)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_tsv(mutations_to_rows(mutations), args.out)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
