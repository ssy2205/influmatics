"""Build an antigenic-coordinate modeling dataset TSV."""

from __future__ import annotations

import argparse

from influenza_protein_dl.dataset import (
    build_coordinate_dataset_from_paths,
    write_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Protein manifest TSV")
    parser.add_argument("--metadata", required=True, help="Metadata TSV")
    parser.add_argument("--labels", required=True, help="Antigenic coordinate label TSV")
    parser.add_argument("--mutation-features", help="Optional mutation feature TSV")
    parser.add_argument("--out", required=True, help="Output dataset TSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_coordinate_dataset_from_paths(
        manifest_path=args.manifest,
        metadata_path=args.metadata,
        labels_path=args.labels,
        mutation_features_path=args.mutation_features,
    )
    for warning in result.warnings:
        print(f"[WARN] {warning}")
    write_dataset(result.frame, args.out)
    print(f"Wrote {len(result.frame)} labeled rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
