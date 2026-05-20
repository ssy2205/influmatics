"""Command line interface for Influmatics."""

from __future__ import annotations

import argparse
from pathlib import Path

from .clade import clade_assignments_to_rows, parse_nextclade_tsv, run_nextclade
from .io import read_sequences, write_tsv
from .mutations import call_mutations_for_alignment, mutations_to_rows
from .numbering import build_numbering_map, numbering_rows_to_tsv, read_numbering_table
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

    numbering_parser = subparsers.add_parser(
        "numbering-map",
        help="Map numbering table positions onto an aligned reference",
    )
    numbering_parser.add_argument("alignment", help="Aligned FASTA containing the reference")
    numbering_parser.add_argument("--reference-id", required=True)
    numbering_parser.add_argument("--numbering-table", required=True)
    numbering_parser.add_argument("--out", required=True, help="Output numbering map TSV")
    numbering_parser.add_argument("--scheme")
    numbering_parser.add_argument("--gene")

    clade_parser = subparsers.add_parser("clade", help="Run Nextclade or parse Nextclade TSV")
    clade_parser.add_argument("--input-fasta", help="Input FASTA for Nextclade")
    clade_parser.add_argument("--nextclade-tsv", help="Existing Nextclade TSV to parse")
    clade_parser.add_argument("--dataset", default="", help="Nextclade dataset name")
    clade_parser.add_argument("--outdir", help="Nextclade output directory")
    clade_parser.add_argument("--out", required=True, help="Output clade summary TSV")

    mutation_parser = subparsers.add_parser("mutations", help="Call mutations from an aligned FASTA")
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

    if args.command == "clade":
        nextclade_tsv = args.nextclade_tsv
        if args.input_fasta:
            if not args.outdir:
                parser.error("--outdir is required when --input-fasta is provided")
            run_nextclade(
                args.input_fasta,
                args.outdir,
                dataset=args.dataset or None,
            )
            nextclade_tsv = str(Path(args.outdir) / "nextclade.tsv")
        if not nextclade_tsv:
            parser.error("Provide --input-fasta or --nextclade-tsv")
        assignments = parse_nextclade_tsv(nextclade_tsv, dataset=args.dataset)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_tsv(clade_assignments_to_rows(assignments), args.out)
        return 0

    if args.command == "mutations":
        records = read_sequences(args.alignment)
        references = [
            record
            for record in records
            if record.seq_id == args.reference_id or record.norm_id == args.reference_id
        ]
        if not references:
            parser.error(f"Reference id was not found in alignment: {args.reference_id}")
        if len(references) > 1:
            parser.error(f"Reference id is not unique in alignment: {args.reference_id}")
        entries = read_numbering_table(args.numbering_table)
        rows = build_numbering_map(
            references[0].sequence,
            entries,
            scheme=args.scheme,
            gene=args.gene,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_tsv(numbering_rows_to_tsv(rows), args.out)
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
