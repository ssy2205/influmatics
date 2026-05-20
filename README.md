# Influmatics

Program for Antigenic Variation Analysis of Seasonal Influenza Virus.

Influenza virus sequence analysis program for collaborative development, with future room for PLM-based viral fitness and mutation prediction workflows.

Influmatics is being organized as a collaborative Python package for influenza sequence analysis. The near-term MVP focuses on FASTA input and reproducible local analysis:

```text
FASTA input
-> QC
-> MAFFT alignment
-> mutation table
-> Nextclade clade annotation
-> antigenic-site mutation detection
-> antiviral marker scan
-> basic report / Streamlit UI
```

ONT FASTQ consensus generation, HyPhy selection analysis, phylogeography, and antigenic cartography are planned as later modules.

## Repository Layout

```text
influmatics/              Python package under active development
data/markers/            Small public marker/site definition tables
data/references/         Placeholder for redistributable references only
docs/                    Roadmap and developer notes
legacy/original_scripts/ Original prototype scripts, preserved for reference
scripts/                 Utility entry points and migration helpers
tests/                   Unit tests for stable package behavior
```

## Install for Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

External command line tools are optional at install time but required by specific modules:

- `mafft` for alignment
- `nextclade` for clade assignment
- `blastn` and local BLAST databases for subtype/source screening
- `minimap2`, `samtools`, `medaka_consensus`, `seqkit` for ONT consensus
- `hyphy` for FEL/MEME selection analyses

## Quick Start

Run basic input QC:

```bash
influmatics qc samples.fasta --out results/qc_summary.tsv
```

Create a mutation table from an aligned reference and sample FASTA:

```bash
influmatics mutations aligned.fasta --reference-id reference --out results/mutations.tsv
```

Translate nucleotide mutations into amino-acid mutations (required before
antigenic-site / antiviral-resistance scanning):

```bash
influmatics translate \
  --alignment aligned.fasta --reference-id reference \
  --cds-start 1 --out results/aa_mutations.tsv
```

The output TSV is stamped with `coordinate_space=aa`, which the antigenic
and resistance scanners check before consuming the table.

## Data Policy

Do not commit restricted sequence datasets, especially GISAID-derived FASTA or metadata. Keep private inputs in `data/private/` or outside the repository.

## Collaboration

Development uses GitHub Flow: work on focused branches, open pull requests, and require at least one teammate review before merging to `main`.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) for the full team workflow.

## Current Status

This repository is not yet a finished application. It is a cleaned project skeleton plus preserved prototypes. The highest-priority engineering work is input validation, numbering/coordinate mapping, alignment and mutation parsing, and decomposing the legacy prototype into tested modules.
