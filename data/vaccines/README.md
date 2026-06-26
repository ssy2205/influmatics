# Vaccine Panels

Place only redistributable vaccine-strain FASTA files here.

The legacy H3N2 HA analysis script looks for this default panel:

```text
data/vaccines/h3n2_ha_vaccine_panel.fasta
```

If that file is present, it is used automatically when `--vaccine` is omitted.
If it is absent, the script falls back to the bundled reference sequence and
records `vaccine_source=reference_fallback` in `run_manifest.json`.

Do not commit GISAID-derived or otherwise restricted sequences. Curate this
panel only from sources that permit redistribution, and keep a companion note
with strain names, accessions, source database, and retrieval date.
