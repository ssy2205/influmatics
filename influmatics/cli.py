"""Command line interface for Influmatics."""

from __future__ import annotations

import argparse
from pathlib import Path

from .alignment import AlignmentError, run_mafft
from .antigenic import (
    antigenic_hits_to_rows,
    read_antigenic_sites,
    read_mutation_rows as read_antigenic_mutation_rows,
    scan_antigenic_sites,
)
from .clade import (
    aa_mutations_to_rows,
    clade_assignments_to_rows,
    parse_nextclade_aa_mutations,
    parse_nextclade_tsv,
    run_nextclade,
)
from .io import read_sequences, write_tsv
from .mutations import call_mutations_for_alignment, mutations_to_rows
from .numbering import build_numbering_map, numbering_rows_to_tsv, read_numbering_table
from .qc import assess_sequences, qc_to_rows
from .report import build_tsv_report, read_tsv, write_basic_html
from .resistance import (
    read_antiviral_markers,
    read_mutation_rows as read_resistance_mutation_rows,
    resistance_hits_to_rows,
    scan_resistance_markers,
)
from .translate import add_translate_subcommand, run_translate_cli
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

    align_parser = subparsers.add_parser("align", help="Align FASTA sequences with MAFFT")
    align_parser.add_argument("input", help="Input FASTA")
    align_parser.add_argument("--out", required=True, help="Output aligned FASTA")
    align_parser.add_argument("--threads", type=int, default=1)
    align_parser.add_argument("--no-auto", action="store_true", help="Disable MAFFT --auto")
    align_parser.add_argument(
        "--reorder",
        action="store_true",
        help="Allow MAFFT to reorder records",
    )

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
    clade_parser.add_argument(
        "--aa-out",
        help=(
            "Optional path to also write an amino-acid mutation TSV parsed from "
            "the Nextclade aaSubstitutions/aaDeletions columns (coordinate_space=aa, "
            "ready for the antigenic and resistance scanners)."
        ),
    )
    clade_parser.add_argument(
        "--no-aa-deletions",
        action="store_true",
        help="When writing --aa-out, skip aaDeletions and keep only substitutions.",
    )

    resistance_parser = subparsers.add_parser("resistance", help="Scan antiviral marker hits")
    resistance_parser.add_argument("mutations", help="Mutation TSV")
    resistance_parser.add_argument("--markers", required=True, help="Antiviral marker TSV")
    resistance_parser.add_argument("--out", required=True, help="Output resistance hit TSV")
    resistance_parser.add_argument("--gene")
    resistance_parser.add_argument("--subtype")

    antigenic_parser = subparsers.add_parser("antigenic", help="Scan antigenic-site mutations")
    antigenic_parser.add_argument("mutations", help="Mutation TSV")
    antigenic_parser.add_argument("--sites", required=True, help="Antigenic site JSON")
    antigenic_parser.add_argument("--out", required=True, help="Output antigenic hit TSV")

    report_parser = subparsers.add_parser("report", help="Build a basic HTML report from TSVs")
    report_parser.add_argument("--title", default="Influmatics Report")
    report_parser.add_argument("--section", action="append", required=True, help="Section as Name:path.tsv")
    report_parser.add_argument("--max-rows", type=int, default=50)
    report_parser.add_argument("--out", required=True, help="Output HTML path")

    mutation_parser = subparsers.add_parser(
        "mutations",
        help="Call mutations from an aligned FASTA",
    )
    mutation_parser.add_argument("alignment", help="Aligned FASTA")
    mutation_parser.add_argument("--reference-id", required=True)
    mutation_parser.add_argument("--out", required=True, help="Output mutation TSV")

    web_parser = subparsers.add_parser(
        "web",
        help="Launch the Streamlit web app in a browser",
    )
    web_parser.add_argument("--port", type=int, default=8501, help="Port to serve on")
    web_parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a browser automatically (e.g. on a remote server)",
    )

    add_translate_subcommand(subparsers)

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

    if args.command == "align":
        try:
            run_mafft(
                args.input,
                args.out,
                threads=args.threads,
                auto=not args.no_auto,
                reorder=args.reorder,
            )
        except (AlignmentError, ValueError) as exc:
            parser.error(str(exc))
        return 0

    if args.command == "numbering-map":
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
        if args.aa_out:
            aa_mutations = parse_nextclade_aa_mutations(
                nextclade_tsv,
                include_deletions=not args.no_aa_deletions,
            )
            Path(args.aa_out).parent.mkdir(parents=True, exist_ok=True)
            write_tsv(aa_mutations_to_rows(aa_mutations), args.aa_out)
        return 0

    if args.command == "resistance":
        mutation_rows = read_resistance_mutation_rows(args.mutations)
        markers = read_antiviral_markers(args.markers)
        hits = scan_resistance_markers(
            mutation_rows,
            markers,
            gene=args.gene,
            subtype=args.subtype,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_tsv(resistance_hits_to_rows(hits), args.out)
        return 0

    if args.command == "antigenic":
        mutation_rows = read_antigenic_mutation_rows(args.mutations)
        definition = read_antigenic_sites(args.sites)
        hits = scan_antigenic_sites(mutation_rows, definition)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_tsv(antigenic_hits_to_rows(hits), args.out)
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
        records = require_valid_sequence_input(args.alignment, allow_duplicate_ids=False)
        mutations = call_mutations_for_alignment(records, args.reference_id)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_tsv(mutations_to_rows(mutations), args.out)
        return 0

    if args.command == "web":
        from .web import launch

        try:
            return launch(port=args.port, headless=args.headless)
        except RuntimeError as exc:
            parser.error(str(exc))

    if args.command == "translate":
        return run_translate_cli(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
