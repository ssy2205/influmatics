"""Validate protein-modeling manifest, metadata, and label TSV files."""

from __future__ import annotations

import argparse

from influenza_protein_dl.schema import (
    read_tsv,
    validate_coordinate_labels,
    validate_manifest,
    validate_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Protein manifest TSV")
    parser.add_argument("--metadata", help="Metadata TSV")
    parser.add_argument("--labels", help="Antigenic coordinate label TSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not any([args.manifest, args.metadata, args.labels]):
        raise SystemExit("Provide at least one of --manifest, --metadata, or --labels")

    ok = True
    if args.manifest:
        ok = _report("manifest", validate_manifest(read_tsv(args.manifest))) and ok
    if args.metadata:
        ok = _report("metadata", validate_metadata(read_tsv(args.metadata))) and ok
    if args.labels:
        ok = _report("labels", validate_coordinate_labels(read_tsv(args.labels))) and ok
    return 0 if ok else 1


def _report(name: str, result) -> bool:
    for warning in result.warnings:
        print(f"[WARN] {name}: {warning}")
    for error in result.errors:
        print(f"[ERR] {name}: {error}")
    if result.ok:
        print(f"[OK] {name}")
    return result.ok


if __name__ == "__main__":
    raise SystemExit(main())
