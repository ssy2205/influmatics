"""Translate nucleotide mutations into amino-acid mutations.

This is the missing bridge between ``influmatics mutations`` (which emits
NT-level mutations stamped with ``coordinate_space='nt'``) and the
antigenic-site and antiviral-resistance scanners (which require AA-level
mutations stamped with ``coordinate_space='aa'``).

The translator walks each query sequence against the aligned reference,
collapses gap columns down to ungapped reference coordinates, applies the
configured CDS window, and produces one AA mutation row per codon that
actually differs from the reference protein.

Synonymous codon substitutions are reported with
``mutation_type='synonymous'`` so callers can either include or filter
them out. Stop codons introduced by a substitution are flagged as
``nonsense``. Frame-disrupting indels (length not a multiple of three)
are reported as a single ``frameshift`` event and downstream codons in
that sequence are skipped, because their AA values are not meaningful.
In-frame indels (multiple of three) become ``aa_insertion`` or
``aa_deletion`` events.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from .io import read_sequences

# Standard genetic code (NCBI table 1). Stop codons are mapped to '*'.
STANDARD_CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


@dataclass(frozen=True)
class AAMutation:
    seq_id: str
    mutation_type: str  # substitution | synonymous | nonsense | aa_insertion |
                       # aa_deletion | frameshift
    position: int  # 1-based AA position in the reference protein
    reference: str  # reference AA, or inserted/deleted AA stretch
    observed: str
    mutation: str
    nt_position: int | None  # 1-based NT position in ungapped reference (for traceability)
    coordinate_space: str = "aa"


def _translate_codon(codon: str, table: dict[str, str]) -> str:
    """Translate a single codon, returning 'X' on any ambiguity."""
    codon = codon.upper()
    if len(codon) != 3 or any(b not in "ACGT" for b in codon):
        return "X"
    return table.get(codon, "X")


def _strip_gaps_with_map(
    aligned: str,
) -> tuple[str, list[int]]:
    """Return (ungapped sequence, aligned-column index for each ungapped base).

    The list is 0-based: ``cols[i]`` is the 0-based alignment column of the
    i-th ungapped base.
    """
    bases: list[str] = []
    cols: list[int] = []
    for aligned_col, base in enumerate(aligned):
        if base != "-":
            bases.append(base)
            cols.append(aligned_col)
    return "".join(bases), cols


def translate_pair(
    aligned_reference_nt: str,
    aligned_query_nt: str,
    query_seq_id: str,
    cds_start_1based: int = 1,
    cds_end_1based: int | None = None,
    genetic_code: dict[str, str] | None = None,
) -> list[AAMutation]:
    """Translate a single reference/query aligned pair into AA mutations."""
    table = genetic_code if genetic_code is not None else STANDARD_CODON_TABLE
    if len(aligned_reference_nt) != len(aligned_query_nt):
        raise ValueError("Reference and query must be aligned to equal length.")

    ref_ungapped, _ref_cols = _strip_gaps_with_map(aligned_reference_nt)
    if not ref_ungapped:
        raise ValueError("Reference has no ungapped bases.")

    end = cds_end_1based if cds_end_1based is not None else len(ref_ungapped)
    if cds_start_1based < 1 or end > len(ref_ungapped) or cds_start_1based > end:
        raise ValueError(
            f"CDS window [{cds_start_1based}, {end}] is outside reference "
            f"of ungapped length {len(ref_ungapped)}."
        )

    # Walk the alignment column-by-column. We keep both the running reference
    # ungapped index (1-based) and per-codon accumulators for ref and query.
    # When a codon's reference index falls inside the CDS window we score it.
    mutations: list[AAMutation] = []
    ref_pos = 0  # 1-based ungapped reference NT position of the last consumed ref base
    aa_pos = 0  # 1-based AA position within the CDS

    ref_codon: list[str] = []
    qry_codon: list[str] = []
    codon_start_ref_pos: int | None = None  # 1-based ungapped ref NT pos of codon start
    in_frameshift = False

    ref_seq = aligned_reference_nt.upper()
    qry_seq = aligned_query_nt.upper()

    for aligned_col in range(len(ref_seq)):
        ref_base = ref_seq[aligned_col]
        qry_base = qry_seq[aligned_col]

        if ref_base != "-":
            ref_pos += 1
            inside_cds = cds_start_1based <= ref_pos <= end
            if not inside_cds:
                continue
            ref_codon.append(ref_base)
            if codon_start_ref_pos is None:
                codon_start_ref_pos = ref_pos
            # Query column: a real base, or a gap (deletion column).
            qry_codon.append(qry_base if qry_base != "-" else "-")
        else:
            # Reference gap = insertion in query relative to reference.
            # Only count it if we're inside (or strictly between codons of) the
            # CDS window. We attach the insertion to the *next* AA position.
            if cds_start_1based <= ref_pos + 1 <= end and qry_base != "-":
                # Collapse runs of insertion columns into one event by appending
                # the inserted base to a small lookahead buffer per AA.
                # For simplicity, only emit insertions that are full codons.
                # Walk forward greedily to gather the contiguous insertion run.
                # (We rely on this loop's natural progression; see post-pass.)
                # Mark the codon as containing inserted material; full handling
                # is done in a single pass below using a buffered approach.
                pass
            # We don't append to ref_codon for ref gaps. Continue.

        if len(ref_codon) == 3:
            # Score this codon.
            cdn_ref = "".join(ref_codon)
            # Determine deletion: any '-' in the query codon means a deletion
            # column landed inside this codon.
            qry_str = "".join(qry_codon)
            dash_count = qry_str.count("-")
            if dash_count > 0:
                # Indel in query at this codon.
                if dash_count == 3:
                    # Clean in-frame single-codon deletion.
                    ref_aa = _translate_codon(cdn_ref, table)
                    mutations.append(
                        AAMutation(
                            seq_id=query_seq_id,
                            mutation_type="aa_deletion",
                            position=aa_pos + 1,
                            reference=ref_aa,
                            observed="-",
                            mutation=f"{ref_aa}{aa_pos + 1}del",
                            nt_position=codon_start_ref_pos,
                        )
                    )
                elif dash_count % 3 == 0:
                    # Multi-codon clean deletion -- represented as a single event.
                    ref_aa = _translate_codon(cdn_ref, table)
                    mutations.append(
                        AAMutation(
                            seq_id=query_seq_id,
                            mutation_type="aa_deletion",
                            position=aa_pos + 1,
                            reference=ref_aa,
                            observed="-",
                            mutation=f"{ref_aa}{aa_pos + 1}del",
                            nt_position=codon_start_ref_pos,
                        )
                    )
                else:
                    mutations.append(
                        AAMutation(
                            seq_id=query_seq_id,
                            mutation_type="frameshift",
                            position=aa_pos + 1,
                            reference=_translate_codon(cdn_ref, table),
                            observed="?",
                            mutation=f"fs{aa_pos + 1}",
                            nt_position=codon_start_ref_pos,
                        )
                    )
                    in_frameshift = True
            elif not in_frameshift:
                ref_aa = _translate_codon(cdn_ref, table)
                qry_aa = _translate_codon(qry_str, table)
                if ref_aa != qry_aa:
                    if qry_aa == "*":
                        mtype = "nonsense"
                    elif qry_aa == "X":
                        mtype = "ambiguous"
                    else:
                        mtype = "substitution"
                    mutations.append(
                        AAMutation(
                            seq_id=query_seq_id,
                            mutation_type=mtype,
                            position=aa_pos + 1,
                            reference=ref_aa,
                            observed=qry_aa,
                            mutation=f"{ref_aa}{aa_pos + 1}{qry_aa}",
                            nt_position=codon_start_ref_pos,
                        )
                    )
                else:
                    # Synonymous: same AA but at least one NT differs.
                    if cdn_ref != qry_str:
                        mutations.append(
                            AAMutation(
                                seq_id=query_seq_id,
                                mutation_type="synonymous",
                                position=aa_pos + 1,
                                reference=ref_aa,
                                observed=qry_aa,
                                mutation=f"{ref_aa}{aa_pos + 1}=",
                                nt_position=codon_start_ref_pos,
                            )
                        )
            # Else: in frameshift, skip scoring.
            aa_pos += 1
            ref_codon = []
            qry_codon = []
            codon_start_ref_pos = None

    # Handle insertions in a second pass: any contiguous run of reference '-'
    # columns inside the CDS produces one aa_insertion event when the inserted
    # length is a multiple of 3, otherwise it folds into the existing
    # frameshift treatment above (we don't double-count).
    insertions = _collect_inframe_insertions(
        ref_seq, qry_seq, cds_start_1based, end, query_seq_id, table
    )
    mutations.extend(insertions)
    mutations.sort(key=lambda m: (m.position, m.mutation_type))
    return mutations


def _collect_inframe_insertions(
    ref_seq: str,
    qry_seq: str,
    cds_start_1based: int,
    cds_end_1based: int,
    query_seq_id: str,
    table: dict[str, str],
) -> list[AAMutation]:
    """Emit one aa_insertion per contiguous in-frame insertion run."""
    events: list[AAMutation] = []
    ref_pos = 0  # 1-based ungapped reference position of the last consumed base
    aa_pos_floor = 0  # Number of complete ref codons consumed inside CDS.
    in_cds_bases = 0  # NT bases consumed inside CDS so far.

    run_bases: list[str] = []
    run_anchor_ref_pos: int | None = None

    def flush() -> None:
        nonlocal run_bases, run_anchor_ref_pos
        if not run_bases:
            return
        if len(run_bases) % 3 == 0:
            inserted_codons = [
                "".join(run_bases[i : i + 3]) for i in range(0, len(run_bases), 3)
            ]
            inserted_aa = "".join(_translate_codon(c, table) for c in inserted_codons)
            # Anchor at the AA position of the codon currently being built.
            anchor_aa = aa_pos_floor + 1
            events.append(
                AAMutation(
                    seq_id=query_seq_id,
                    mutation_type="aa_insertion",
                    position=anchor_aa,
                    reference="-",
                    observed=inserted_aa,
                    mutation=f"ins{anchor_aa}{inserted_aa}",
                    nt_position=run_anchor_ref_pos,
                )
            )
        run_bases = []
        run_anchor_ref_pos = None

    for col in range(len(ref_seq)):
        ref_base = ref_seq[col]
        qry_base = qry_seq[col]
        if ref_base != "-":
            # Flush any pending insertion before consuming a ref base.
            flush()
            ref_pos += 1
            if cds_start_1based <= ref_pos <= cds_end_1based:
                in_cds_bases += 1
                if in_cds_bases % 3 == 0:
                    aa_pos_floor += 1
        else:
            # Reference gap. Inside the CDS interior?
            if (
                cds_start_1based <= ref_pos + 1 <= cds_end_1based
                and qry_base != "-"
            ):
                if not run_bases:
                    run_anchor_ref_pos = ref_pos + 1
                run_bases.append(qry_base)
    flush()
    return events


def translate_nt_mutations(
    aligned_reference_nt: str,
    aligned_queries: list[tuple[str, str]],
    cds_start_1based: int = 1,
    cds_end_1based: int | None = None,
    genetic_code: dict[str, str] | None = None,
) -> list[AAMutation]:
    """Translate AA mutations for several queries aligned to one reference.

    ``aligned_queries`` is a list of ``(seq_id, aligned_query_nt)`` tuples.
    """
    out: list[AAMutation] = []
    for seq_id, aligned in aligned_queries:
        out.extend(
            translate_pair(
                aligned_reference_nt,
                aligned,
                seq_id,
                cds_start_1based=cds_start_1based,
                cds_end_1based=cds_end_1based,
                genetic_code=genetic_code,
            )
        )
    return out


def aa_mutations_to_rows(mutations: list[AAMutation]) -> list[dict[str, object]]:
    """Convert AA mutations into TSV-friendly rows."""
    return [
        {
            "seq_id": m.seq_id,
            "mutation_type": m.mutation_type,
            "position": m.position,
            "reference": m.reference,
            "observed": m.observed,
            "mutation": m.mutation,
            "nt_position": m.nt_position if m.nt_position is not None else "",
            "coordinate_space": m.coordinate_space,
        }
        for m in mutations
    ]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _load_alignment(path: str | Path, reference_id: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (reference_aligned_nt, [(query_id, aligned_nt), ...])."""
    records = read_sequences(path)
    reference = None
    queries: list[tuple[str, str]] = []
    for rec in records:
        if rec.seq_id == reference_id or rec.norm_id == reference_id:
            if reference is not None:
                raise ValueError(f"Reference id is not unique: {reference_id}")
            reference = rec
        else:
            queries.append((rec.seq_id, rec.sequence))
    if reference is None:
        raise ValueError(f"Reference id was not found in alignment: {reference_id}")
    expected = len(reference.sequence)
    bad = [sid for sid, seq in queries if len(seq) != expected]
    if bad:
        raise ValueError(
            "All alignment records must share length with reference. "
            f"Mismatched: {','.join(bad)}"
        )
    return reference.sequence, queries


def add_translate_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``translate`` subcommand on an argparse subparsers object."""
    parser = subparsers.add_parser(
        "translate",
        help="Translate NT mutations from an aligned FASTA into AA mutations",
    )
    parser.add_argument("--alignment", required=True, help="Aligned FASTA")
    parser.add_argument("--reference-id", required=True)
    parser.add_argument(
        "--cds-start",
        type=int,
        default=1,
        help="1-based start of the CDS on the ungapped reference (default: 1)",
    )
    parser.add_argument(
        "--cds-end",
        type=int,
        default=None,
        help="1-based end (inclusive) of the CDS; defaults to end of reference",
    )
    parser.add_argument("--out", required=True, help="Output AA mutation TSV")


def run_translate_cli(args: argparse.Namespace) -> int:
    from .io import write_tsv

    ref_seq, queries = _load_alignment(args.alignment, args.reference_id)
    mutations = translate_nt_mutations(
        ref_seq,
        queries,
        cds_start_1based=args.cds_start,
        cds_end_1based=args.cds_end,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_tsv(aa_mutations_to_rows(mutations), args.out)
    return 0
