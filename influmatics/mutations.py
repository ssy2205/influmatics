"""Reference-based mutation parsing.

Outputs nucleotide-level mutations by default. The ``coordinate_space`` field
is stamped onto every record (and into the TSV) so downstream scanners
(antigenic-site, antiviral-resistance) can refuse to consume nucleotide-coordinate
data where amino-acid coordinates are required.

Consecutive insertions and deletions are coalesced into single mutation events
to match HGVS-style conventions (e.g. ``ins823GGG`` instead of three separate
``ins823G`` records). Positions that involve ambiguous IUPAC codes (``N`` and
friends) are emitted as ``ambiguous`` events rather than substitutions, since
real-world sequencing data otherwise produces a large amount of meaningless
substitution noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from .io import SeqRecord

# Unambiguous DNA alphabet. Anything outside this set is treated as ambiguous
# and is reported as a distinct mutation_type instead of being silently called
# a substitution.
_UNAMBIGUOUS = frozenset("ACGT-")


@dataclass(frozen=True)
class Mutation:
    seq_id: str
    mutation_type: str
    position: int
    alignment_position: int
    reference_position: int | None
    query_position: int | None
    reference: str
    observed: str
    mutation: str
    # Coordinate space of `position`. "nt" for raw nucleotide alignment
    # coordinates, "aa" for amino-acid coordinates. Downstream scanners check
    # this field before interpreting the data.
    coordinate_space: str = "nt"


def _is_ambiguous(base: str) -> bool:
    return base not in _UNAMBIGUOUS


def call_nt_mutations(
    aligned_reference: SeqRecord, aligned_query: SeqRecord
) -> list[Mutation]:
    """Call nucleotide differences from a pair of aligned sequences.

    Coalesces consecutive insertions and deletions into single events.
    """

    if len(aligned_reference.sequence) != len(aligned_query.sequence):
        raise ValueError(
            "Reference and query sequences must already be aligned to equal length."
        )

    ref_seq = aligned_reference.sequence.upper()
    qry_seq = aligned_query.sequence.upper()

    mutations: list[Mutation] = []
    reference_position = 0
    query_position = 0

    # Run-state for coalescing consecutive indels.
    ins_run: list[str] = []
    ins_run_start_align: int | None = None
    ins_run_anchor_ref: int = 0  # Last reference position before the insertion.
    ins_run_start_query: int | None = None

    del_run: list[str] = []
    del_run_start_align: int | None = None
    del_run_start_ref: int | None = None

    def flush_insertion() -> None:
        nonlocal ins_run, ins_run_start_align, ins_run_start_query
        if not ins_run:
            return
        inserted = "".join(ins_run)
        anchor = ins_run_anchor_ref + 1  # Insertion is "before" this position.
        mutation_name = f"ins{anchor}{inserted}"
        mutations.append(
            Mutation(
                seq_id=aligned_query.seq_id,
                mutation_type="insertion",
                position=anchor,
                alignment_position=ins_run_start_align or 0,
                reference_position=None,
                query_position=ins_run_start_query,
                reference="-",
                observed=inserted,
                mutation=mutation_name,
                coordinate_space="nt",
            )
        )
        ins_run = []
        ins_run_start_align = None
        ins_run_start_query = None

    def flush_deletion() -> None:
        nonlocal del_run, del_run_start_align, del_run_start_ref
        if not del_run:
            return
        deleted = "".join(del_run)
        start = del_run_start_ref or 0
        if len(deleted) == 1:
            mutation_name = f"{deleted}{start}del"
        else:
            end = start + len(deleted) - 1
            mutation_name = f"{deleted}{start}_{end}del"
        mutations.append(
            Mutation(
                seq_id=aligned_query.seq_id,
                mutation_type="deletion",
                position=start,
                alignment_position=del_run_start_align or 0,
                reference_position=start,
                query_position=None,
                reference=deleted,
                observed="-",
                mutation=mutation_name,
                coordinate_space="nt",
            )
        )
        del_run = []
        del_run_start_align = None
        del_run_start_ref = None

    for alignment_position, (ref_base, query_base) in enumerate(
        zip(ref_seq, qry_seq), start=1
    ):
        if ref_base != "-":
            reference_position += 1
        if query_base != "-":
            query_position += 1

        # End any open insertion/deletion run when the column type changes.
        if ref_base != "-" and ins_run:
            flush_insertion()
        if query_base != "-" and del_run:
            flush_deletion()

        if ref_base == query_base:
            continue

        if ref_base == "-":
            # Insertion in query relative to reference. Anchor on the last
            # consumed reference position.
            if not ins_run:
                ins_run_start_align = alignment_position
                ins_run_anchor_ref = reference_position
                ins_run_start_query = query_position
            ins_run.append(query_base)
            continue

        if query_base == "-":
            if not del_run:
                del_run_start_align = alignment_position
                del_run_start_ref = reference_position
            del_run.append(ref_base)
            continue

        # Both sides are present. Decide ambiguous vs substitution.
        if _is_ambiguous(ref_base) or _is_ambiguous(query_base):
            mutation_type = "ambiguous"
        else:
            mutation_type = "substitution"

        mutation_name = f"{ref_base}{reference_position}{query_base}"
        mutations.append(
            Mutation(
                seq_id=aligned_query.seq_id,
                mutation_type=mutation_type,
                position=reference_position,
                alignment_position=alignment_position,
                reference_position=reference_position,
                query_position=query_position,
                reference=ref_base,
                observed=query_base,
                mutation=mutation_name,
                coordinate_space="nt",
            )
        )

    # Flush any trailing runs.
    flush_insertion()
    flush_deletion()
    return mutations


def call_mutations_for_alignment(
    records: list[SeqRecord], reference_id: str
) -> list[Mutation]:
    """Call mutations for every non-reference record in an aligned FASTA."""

    references = [
        record
        for record in records
        if record.seq_id == reference_id or record.norm_id == reference_id
    ]
    if not references:
        raise ValueError(f"Reference id was not found in alignment: {reference_id}")
    if len(references) > 1:
        raise ValueError(f"Reference id is not unique in alignment: {reference_id}")
    reference = references[0]
    expected_length = len(reference.sequence)
    mismatched = [
        record.seq_id for record in records if len(record.sequence) != expected_length
    ]
    if mismatched:
        raise ValueError(
            "All records in the alignment must have equal length. "
            f"Mismatched records: {','.join(mismatched)}"
        )
    mutations: list[Mutation] = []
    for record in records:
        if record.seq_id == reference.seq_id:
            continue
        mutations.extend(call_nt_mutations(reference, record))
    return mutations


def mutations_to_rows(mutations: list[Mutation]) -> list[dict[str, object]]:
    """Convert mutation calls into TSV-friendly rows."""

    def _coord(value: int | None) -> object:
        return "" if value is None else value

    return [
        {
            "seq_id": mutation.seq_id,
            "mutation_type": mutation.mutation_type,
            "position": mutation.position,
            "alignment_position": mutation.alignment_position,
            "reference_position": _coord(mutation.reference_position),
            "query_position": _coord(mutation.query_position),
            "reference": mutation.reference,
            "observed": mutation.observed,
            "mutation": mutation.mutation,
            "coordinate_space": mutation.coordinate_space,
        }
        for mutation in mutations
    ]
