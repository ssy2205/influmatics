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

Align sequences with MAFFT:

```bash
influmatics align samples.fasta --out results/aligned.fasta --threads 4
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

Alternatively, assign clades with Nextclade and export an amino-acid
mutation table parsed straight from its `aaSubstitutions`/`aaDeletions`
columns (also stamped `coordinate_space=aa`):

```bash
influmatics clade \
  --input-fasta samples.fasta --dataset flu_h3n2_ha --outdir results/nextclade \
  --out results/clades.tsv --aa-out results/aa_mutations.tsv
```

Either AA table then feeds the scanners directly:

```bash
influmatics antigenic results/aa_mutations.tsv \
  --sites data/markers/antigenic_sites_h3n2.json --out results/antigenic_hits.tsv
influmatics resistance results/aa_mutations.tsv \
  --markers data/markers/antiviral_markers.tsv --out results/resistance_hits.tsv
```

Curated marker tables ship in `data/markers/` (H3N2 and H1N1 antigenic
sites, plus NAI/adamantane antiviral markers). See
[docs/marker_curation_report.md](docs/marker_curation_report.md) for sourcing.

### Web app

Run the whole analysis in the browser instead of the command line:

```bash
pip install -e ".[web]"   # one-time: installs Streamlit
influmatics web           # opens http://localhost:8501 in your browser
```

The app has a **QC** tab (upload a FASTA, get the QC table and a download)
and an **Antigenic / Resistance scan** tab (upload an amino-acid mutation
TSV — e.g. from `influmatics translate` or `influmatics clade --aa-out` —
and scan it against a bundled curated marker table). Use
`influmatics web --port 8600` to pick a port or `--headless` on a remote
server.

## Data Policy

Do not commit restricted sequence datasets, especially GISAID-derived FASTA or metadata. Keep private inputs in `data/private/` or outside the repository.

## Collaboration

Development uses GitHub Flow: work on focused branches, open pull requests, and require at least one teammate review before merging to `main`.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) for the full team workflow.

## User Guide

Want the big picture? Open the mobile-friendly project summary (features +
what was built): [docs/project_summary.html](docs/project_summary.html).

New here? Open the easiest, step-by-step web-app walkthrough (written for
absolute beginners): [docs/easy_start.html](docs/easy_start.html).

For the fuller, command-line oriented guide, open
[docs/usage_guide.html](docs/usage_guide.html) in a browser.

For model training decisions, data collection, and server/GPU guidance, open
[docs/deep_learning_training_guide.html](docs/deep_learning_training_guide.html).

For beginner-friendly data collection instructions, including which sites to
use and which files to download, open
[docs/data_collection_guide.html](docs/data_collection_guide.html).

For the frontend/server handoff, including Git clone instructions, local setup,
analysis CLI contract, API suggestions, and a starter prompt for another Codex
session, open [docs/web_handoff.html](docs/web_handoff.html).

For the next product iteration plan focused on built-in background datasets,
automatic TreeTime metadata, QC review, FASTQ mapping, and preview-site UX,
open [docs/ux_pipeline_improvement_plan.html](docs/ux_pipeline_improvement_plan.html).
On the Firebase preview deployment, the same handoff page is served at
`/docs/ux_pipeline_improvement_plan.html`.

For a longer beginner-friendly map of the software concepts needed to build this
project as a web service, open
[docs/software_concepts_guide.html](docs/software_concepts_guide.html).

## Current Status

This repository is not yet a finished application. It is a cleaned project skeleton plus preserved prototypes. The highest-priority engineering work is input validation, numbering/coordinate mapping, alignment and mutation parsing, and decomposing the legacy prototype into tested modules.
