from influenza_protein_dl.sequence_features import (
    aa_composition,
    kmer_counts,
    summarize_protein_sequence,
)


def test_summarize_protein_sequence_counts_ambiguous_and_invalid_characters():
    summary = summarize_protein_sequence("ACDXX*-?")

    assert summary.length == 8
    assert summary.valid_aa_count == 3
    assert summary.ambiguous_aa_count == 2
    assert summary.stop_count == 1
    assert summary.gap_count == 1
    assert summary.invalid_aa_count == 1
    assert summary.invalid_characters == "?"


def test_aa_composition_uses_only_canonical_residues_as_denominator():
    features = aa_composition("AACCXX")

    assert features["aa_A"] == 0.5
    assert features["aa_C"] == 0.5
    assert features["aa_D"] == 0.0


def test_kmer_counts_skips_noncanonical_kmers():
    counts = kmer_counts("ACDEXXACD", k=3)

    assert counts["ACD"] == 2
    assert counts["CDE"] == 1
    assert "EXX" not in counts
