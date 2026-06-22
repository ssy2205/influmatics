"""Streamlit MVP for Influmatics.

The pure helpers below (module listing, bundled-marker lookup, QC and scan
runners) carry the actual analysis logic so they can be unit-tested without a
running Streamlit server. ``main()`` is a thin glue layer that wires file
uploads to those helpers and renders/downloads the results.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .io import SeqRecord, read_sequences
from .qc import assess_sequences, qc_to_rows

# Curated marker tables ship under the repository data directory, one level
# above the package. They are resolved lazily so importing this module never
# depends on the data being present.
_MARKERS_DIR = Path(__file__).resolve().parents[1] / "data" / "markers"


@dataclass(frozen=True)
class WebModule:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class BundledMarkerFile:
    key: str
    label: str
    kind: str  # "antigenic" or "resistance"
    path: Path


def available_modules() -> list[WebModule]:
    """Return MVP modules exposed in the Streamlit app."""

    return [
        WebModule("qc", "QC", "Sequence length, ambiguity, gaps, and invalid characters"),
        WebModule("alignment", "MAFFT alignment", "Multiple sequence alignment with MAFFT"),
        WebModule("mutations", "Mutation table", "Reference-based nucleotide mutation table"),
        WebModule("clade", "Nextclade clade", "Clade summary from Nextclade outputs"),
        WebModule("antigenic", "Antigenic sites", "Position-based antigenic-site mutation scan"),
        WebModule("resistance", "Resistance markers", "Curated antiviral marker scan"),
        WebModule("report", "Report", "Downloadable TSV/HTML result summaries"),
    ]


def selected_module_labels(selected_keys: list[str]) -> list[str]:
    """Return labels for selected module keys."""

    modules = {module.key: module.label for module in available_modules()}
    return [modules[key] for key in selected_keys if key in modules]


def bundled_marker_files() -> list[BundledMarkerFile]:
    """Return the curated marker tables shipped with the project."""

    return [
        BundledMarkerFile(
            "h3n2_sites",
            "H3N2 antigenic sites",
            "antigenic",
            _MARKERS_DIR / "antigenic_sites_h3n2.json",
        ),
        BundledMarkerFile(
            "h1n1_sites",
            "H1N1 antigenic sites",
            "antigenic",
            _MARKERS_DIR / "antigenic_sites_h1n1.json",
        ),
        BundledMarkerFile(
            "antiviral",
            "Antiviral resistance markers",
            "resistance",
            _MARKERS_DIR / "antiviral_markers.tsv",
        ),
    ]


def run_qc_rows(
    records: list[SeqRecord],
    min_length: int = 500,
    max_ambiguous_fraction: float = 0.05,
    max_gap_fraction: float = 0.05,
) -> list[dict[str, object]]:
    """Run QC on parsed records and return TSV-ready rows."""

    results = assess_sequences(
        records,
        min_length=min_length,
        max_ambiguous_fraction=max_ambiguous_fraction,
        max_gap_fraction=max_gap_fraction,
    )
    return qc_to_rows(results)


def scan_aa_table(
    aa_tsv_path: str | Path,
    marker_file: BundledMarkerFile,
) -> list[dict[str, object]]:
    """Scan an amino-acid mutation TSV against a bundled marker file.

    Dispatches on ``marker_file.kind`` so the same entry point drives both the
    antigenic-site and antiviral-resistance scanners.
    """

    if marker_file.kind == "antigenic":
        from .antigenic import (
            antigenic_hits_to_rows,
            read_antigenic_sites,
            read_mutation_rows,
            scan_antigenic_sites,
        )

        rows = read_mutation_rows(aa_tsv_path)
        definition = read_antigenic_sites(marker_file.path)
        return antigenic_hits_to_rows(scan_antigenic_sites(rows, definition))

    if marker_file.kind == "resistance":
        from .resistance import (
            read_antiviral_markers,
            read_mutation_rows,
            resistance_hits_to_rows,
            scan_resistance_markers,
        )

        rows = read_mutation_rows(aa_tsv_path)
        markers = read_antiviral_markers(marker_file.path)
        return resistance_hits_to_rows(scan_resistance_markers(rows, markers))

    raise ValueError(f"Unknown marker file kind: {marker_file.kind}")


def rows_to_tsv_text(rows: list[dict[str, object]]) -> str:
    """Serialize result rows to TSV text for download."""

    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _save_upload_to_tempfile(upload, suffix: str) -> Path:
    """Persist a Streamlit UploadedFile to a temp path for path-based readers."""

    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(upload.getvalue())
    handle.close()
    return Path(handle.name)


def streamlit_command(port: int = 8501, headless: bool = False) -> list[str]:
    """Build the ``streamlit run`` command that serves this app.

    Invoked via ``python -m streamlit`` (not the bare ``streamlit`` script)
    so it works regardless of whether the console entry point is on PATH.
    """

    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        __file__,
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
    ]


def launch(port: int = 8501, headless: bool = False) -> int:
    """Launch the Streamlit app in a browser and block until it exits.

    Returns the Streamlit process exit code. Raises a clear error if
    Streamlit is not installed (it lives in the optional ``web`` extra).
    """

    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError(
            "Streamlit is not installed. Install the web extra with "
            "`pip install -e '.[web]'` (or `pip install streamlit`)."
        )
    return subprocess.run(streamlit_command(port=port, headless=headless)).returncode


def main() -> None:
    """Run the Streamlit app."""

    import streamlit as st

    st.set_page_config(page_title="Influmatics", layout="wide")
    st.title("Influmatics")
    st.caption("Program for Antigenic Variation Analysis of Seasonal Influenza Virus")

    qc_tab, scan_tab = st.tabs(["QC", "Antigenic / Resistance scan"])

    with qc_tab:
        st.subheader("Sequence QC")
        # Accept gzipped FASTA too; public-DB downloads are routinely .gz.
        fasta = st.file_uploader("FASTA input", type=["fa", "fasta", "fna", "fas", "gz"])
        min_length = st.number_input("Minimum length", min_value=0, value=500, step=50)
        if fasta is not None:
            suffix = "".join(Path(fasta.name).suffixes) or ".fasta"
            tmp_path = _save_upload_to_tempfile(fasta, suffix)
            try:
                records = read_sequences(tmp_path)
                rows = run_qc_rows(records, min_length=int(min_length))
            except (ValueError, OSError) as exc:
                st.error(f"Could not run QC: {exc}")
            else:
                st.dataframe(rows, use_container_width=True)
                st.download_button(
                    "Download QC TSV",
                    rows_to_tsv_text(rows),
                    file_name="qc_summary.tsv",
                    mime="text/tab-separated-values",
                )
            finally:
                tmp_path.unlink(missing_ok=True)

    with scan_tab:
        st.subheader("Scan an amino-acid mutation table")
        st.caption(
            "Upload an AA mutation TSV (coordinate_space=aa), e.g. from "
            "`influmatics translate` or `influmatics clade --aa-out`."
        )
        aa_tsv = st.file_uploader("AA mutation TSV", type=["tsv"], key="aa_tsv")
        marker_files = bundled_marker_files()
        choice = st.selectbox(
            "Marker table",
            marker_files,
            format_func=lambda marker: marker.label,
        )
        if aa_tsv is not None and choice is not None:
            tmp_path = _save_upload_to_tempfile(aa_tsv, ".tsv")
            try:
                hits = scan_aa_table(tmp_path, choice)
            except (ValueError, OSError) as exc:
                st.error(f"Could not run scan: {exc}")
            else:
                if hits:
                    st.dataframe(hits, use_container_width=True)
                    st.download_button(
                        "Download hits TSV",
                        rows_to_tsv_text(hits),
                        file_name=f"{choice.key}_hits.tsv",
                        mime="text/tab-separated-values",
                    )
                else:
                    st.info("No hits for the selected marker table.")
            finally:
                tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
