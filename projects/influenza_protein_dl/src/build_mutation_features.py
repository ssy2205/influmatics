"""Build sequence-level features from an amino-acid mutation TSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from influenza_protein_dl.mutation_features import (
    build_mutation_feature_table,
    read_antigenic_site_json,
    read_mutation_tsv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aa-mutations", required=True, help="AA mutation TSV")
    parser.add_argument("--antigenic-sites", help="Optional antigenic-site JSON")
    parser.add_argument("--out", required=True, help="Output feature TSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sites = read_antigenic_site_json(args.antigenic_sites) if args.antigenic_sites else None
    features = build_mutation_feature_table(read_mutation_tsv(args.aa_mutations), sites)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, sep="\t", index=False)
    print(f"Wrote {len(features)} mutation feature rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
