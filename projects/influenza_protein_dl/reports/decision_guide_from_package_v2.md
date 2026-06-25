# Decision Guide Imported From Package v2

This note applies the user decisions and curation guidance from
`influenza_analysis_package_v2/guidelines.md` to the protein deep-learning
project.

## Applied Decisions

- Primary target: start with H3N2 HA1 antigenic prediction rather than NA/M2
  resistance prediction.
- First supervised target: antigenic map coordinates or an antigenic novelty
  score, depending on available labels.
- Data policy: use public NCBI/IRD or openly licensed supplementary data for
  committed examples; keep GISAID and other restricted sequence/metadata files
  outside git or in ignored private paths.
- Reference context: use A/Aichi/2/1968 for H3 numbering context and
  A/California/07/2009 for H1N1pdm09 context.
- Modeling path: CPU-friendly baseline features first, frozen PLM embeddings
  second, fine-tuning only after labels and compute are available.
- Biological sanity checks: H3 key antigenic residues, H1 antigenic sites,
  glycosylation motif changes, receptor-binding-site proximity, and
  clade/year ablation.

## Files Applied

- Public reference FASTAs were copied into `data/references/`.
- H3N2 and H1N1 antigenic-site JSON files are available under `data/markers/`.
- NA/M2 antiviral marker TSV entries are available in
  `data/markers/antiviral_markers.tsv`.
- The deep-learning scaffold in `projects/influenza_protein_dl/` can now use
  those marker definitions to create mutation-derived baseline features.

## Citation Cleanup Note

The package v2 markdown files contain inline citation placeholders copied from
an external drafting environment. They are useful as internal trace notes but
should be replaced with formal bibliography entries before any manuscript,
poster, or public report.
