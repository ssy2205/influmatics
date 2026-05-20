"""Streamlit MVP for Influmatics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WebModule:
    key: str
    label: str
    description: str


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


def main() -> None:
    """Run the Streamlit app."""

    import streamlit as st

    st.set_page_config(page_title="Influmatics", layout="wide")
    st.title("Influmatics")
    st.caption("Program for Antigenic Variation Analysis of Seasonal Influenza Virus")

    left, right = st.columns([2, 1])
    with left:
        fasta = st.file_uploader("FASTA input", type=["fa", "fasta", "fna", "fas"])
        metadata = st.file_uploader("Metadata TSV or CSV", type=["tsv", "csv"])
        result_tsv = st.file_uploader("Existing result TSV", type=["tsv"])

    with right:
        st.subheader("Modules")
        selected = []
        for module in available_modules():
            if st.checkbox(module.label, value=module.key in {"qc", "mutations", "report"}):
                selected.append(module.key)
                st.caption(module.description)

    st.subheader("Run plan")
    if selected:
        for label in selected_module_labels(selected):
            st.write(f"- {label}")
    else:
        st.write("No modules selected.")

    st.subheader("Inputs")
    st.write(
        {
            "fasta": fasta.name if fasta else None,
            "metadata": metadata.name if metadata else None,
            "result_tsv": result_tsv.name if result_tsv else None,
        }
    )

    st.info(
        "This MVP scaffold captures inputs and module choices. "
        "Execution wiring will be added after the CLI PRs are merged."
    )


if __name__ == "__main__":
    main()
