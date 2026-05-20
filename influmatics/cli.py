"""Command line interface for Influmatics."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import read_sequences, write_tsv
from .mutations import call_mutations_for_alignment, mutations_to_rows
from .qc import assess_sequences, qc_to_rows
from .report import build_tsv_report, read_tsv, write_basic_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="influmatics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    qc_parser = subparsers.add_parser("qc", help="Run basic sequence QC")
    qc_parser.add_argument("input", help="FASTA, FASTQ, CSV, or TSV input")
    qc_parser.add_argument("--out", required=True, help="Output QC summary TSV")
    qc_parser.add_argument("--min-length", type=int, default=500)
    qc_parser.add_argument("--max-ambiguous-fraction", type=float, default=0.05)

    report_parser = subparsers.add_parser("report", help="Build a basic HTML report from TSVs")
    report_parser.add_argument("--title", default="Influmatics Report")
    report_parser.add_argument("--section", action="append", required=True, help="Section as Name:path.tsv")
    report_parser.add_argument("--max-rows", type=int, default=50)
    report_parser.add_argument("--out", required=True, help="Output HTML path")

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

    if args.command == "report":
        sections = []
        for section in args.section:
            if ":" not in section:
                parser.error("--section must use Name:path.tsv")
            name, path = section.split(":", 1)
            sections.append((name, read_tsv(path)))
        body = build_tsv_report(args.title, sections, max_rows=args.max_rows)
        write_basic_html(args.title, body, args.out)
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
