# Roadmap

## MVP Scope

Target: stable FASTA-based analysis before expanding to ONT and advanced methods.

```text
FASTA input
-> QC
-> MAFFT alignment
-> mutation table
-> Nextclade clade
-> antigenic-site mutation detection
-> antiviral marker scan
-> HTML / Streamlit report
```

## First Five Issues

1. Implement and test `influmatics.qc`
2. Add a robust MAFFT wrapper
3. Build reference-based mutation parser
4. Build H1/H3/NA numbering mapper
5. Add Nextclade CLI wrapper and output parser

## Later Modules

- ONT FASTQ consensus wrapper with QC reports
- Local BLAST subtype/source screening
- Unified R tree visualizer
- HyPhy FEL/MEME wrapper
- Metadata phylogeography and later Treetime mugration
- HI-table-based antigenic cartography

## Key Risks

- Incorrect H3/H1/NA numbering invalidates antigenic-site and resistance calls.
- GISAID-derived sequence data must not be committed.
- UPGMA and simplified dN/dS summaries are exploratory, not final evidence.
- Antigenic cartography requires HI tables; sequence-only site scans are not cartography.
