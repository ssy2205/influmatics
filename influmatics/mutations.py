"""Reference-based mutation parsing."""

from __future__ import annotations

from dataclasses import dataclass

from .io import SeqRecord


@dataclass(frozen=True)
class Mutation:
    seq_id: str
    position: int
    reference: str
    observed: str
    mutation: str


def call_nt_mutations(aligned_reference: SeqRecord, aligned_query: SeqRecord) -> list[Mutation]:
    """Call nucleotide differences from a pair of aligned sequences."""

    if len(aligned_reference.sequence) != len(aligned_query.sequence):
        raise ValueError("Reference and query sequences must already be aligned to equal length.")

    mutations: list[Mutation] = []
    reference_position = 0
    for ref_base, query_base in zip(
        aligned_reference.sequence.upper(),
        aligned_query.sequence.upper(),
    ):
        if ref_base != "-":
            reference_position += 1
        if ref_base == query_base:
            continue
        if ref_base == "-":
            mutation_name = f"ins{reference_position + 1}{query_base}"
            position = reference_position + 1
        elif query_base == "-":
            mutation_name = f"{ref_base}{reference_position}del"
            position = reference_position
        else:
            mutation_name = f"{ref_base}{reference_position}{query_base}"
            position = reference_position
        mutations.append(
            Mutation(
                seq_id=aligned_query.seq_id,
                position=position,
                reference=ref_base,
                observed=query_base,
                mutation=mutation_name,
            )
        )
    return mutations


def call_mutations_for_alignment(records: list[SeqRecord], reference_id: str) -> list[Mutation]:
    """Call mutations for every non-reference record in an aligned FASTA."""

    references = [
        record
        for record in records
        if record.seq_id == reference_id or record.norm_id == reference_id
    ]
    if not references:
        raise ValueError(f"Reference id was not found in alignment: {reference_id}")
    reference = references[0]
    mutations: list[Mutation] = []
    for record in records:
        if record is reference:
            continue
        mutations.extend(call_nt_mutations(reference, record))
    return mutations


def mutations_to_rows(mutations: list[Mutation]) -> list[dict[str, object]]:
    """Convert mutation calls into TSV-friendly rows."""

    return [
        {
            "seq_id": mutation.seq_id,
            "position": mutation.position,
            "reference": mutation.reference,
            "observed": mutation.observed,
            "mutation": mutation.mutation,
        }
        for mutation in mutations
    ]
