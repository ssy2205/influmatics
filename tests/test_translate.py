"""Tests for the NT->AA translator."""

from __future__ import annotations

import pytest

from influmatics.translate import (
    STANDARD_CODON_TABLE,
    aa_mutations_to_rows,
    translate_pair,
)


def test_codon_table_spot_checks():
    # Methionine (start), tryptophan, three stop codons.
    assert STANDARD_CODON_TABLE["ATG"] == "M"
    assert STANDARD_CODON_TABLE["TGG"] == "W"
    assert STANDARD_CODON_TABLE["TAA"] == "*"
    assert STANDARD_CODON_TABLE["TAG"] == "*"
    assert STANDARD_CODON_TABLE["TGA"] == "*"


def test_synonymous_substitution_reported_separately():
    # CTG -> CTA: both encode Leucine. Synonymous, not silently dropped.
    ref = "ATGCTGTAA"  # M L *
    qry = "ATGCTATAA"  # M L *
    muts = translate_pair(ref, qry, "sample")
    types = {m.mutation_type for m in muts}
    assert types == {"synonymous"}
    assert muts[0].mutation == "L2="
    assert muts[0].position == 2


def test_nonsynonymous_substitution_h274y_style():
    # Codon 2: CAT (H) -> TAT (Y).
    ref = "ATGCATTGG"  # M H W
    qry = "ATGTATTGG"  # M Y W
    muts = translate_pair(ref, qry, "sample")
    subs = [m for m in muts if m.mutation_type == "substitution"]
    assert len(subs) == 1
    assert subs[0].mutation == "H2Y"
    assert subs[0].position == 2
    assert subs[0].nt_position == 4  # codon 2 starts at NT 4


def test_stop_codon_introduction_is_nonsense():
    # Codon 2: CAA (Q) -> TAA (*)
    ref = "ATGCAATGG"  # M Q W
    qry = "ATGTAATGG"  # M * W
    muts = translate_pair(ref, qry, "sample")
    nonsense = [m for m in muts if m.mutation_type == "nonsense"]
    assert len(nonsense) == 1
    assert nonsense[0].mutation == "Q2*"


def test_inframe_codon_deletion():
    # Ref:   ATG CAT TGG  (M H W)
    # Query: ATG --- TGG  (M - W)  -> H2del
    ref = "ATGCATTGG"
    qry = "ATG---TGG"
    muts = translate_pair(ref, qry, "sample")
    dels = [m for m in muts if m.mutation_type == "aa_deletion"]
    assert len(dels) == 1
    assert dels[0].mutation == "H2del"
    assert dels[0].position == 2


def test_inframe_codon_insertion():
    # Ref:   ATG --- TGG  (M W after collapsing the gap)
    # Query: ATG CAT TGG  -> insertion of H at AA position 2
    ref = "ATG---TGG"
    qry = "ATGCATTGG"
    muts = translate_pair(ref, qry, "sample")
    ins = [m for m in muts if m.mutation_type == "aa_insertion"]
    assert len(ins) == 1
    assert ins[0].observed == "H"
    # Insertion anchors at the next AA in the reference protein (here W=2).
    assert ins[0].position == 2


def test_single_base_indel_is_frameshift():
    # One-base deletion in query at codon 2 disrupts the frame.
    ref = "ATGCATTGG"
    qry = "ATGC-TTGG"
    muts = translate_pair(ref, qry, "sample")
    fs = [m for m in muts if m.mutation_type == "frameshift"]
    assert len(fs) >= 1
    assert fs[0].position == 2


def test_output_tsv_rows_carry_coordinate_space():
    ref = "ATGCATTGG"
    qry = "ATGTATTGG"
    rows = aa_mutations_to_rows(translate_pair(ref, qry, "sample"))
    assert all(row["coordinate_space"] == "aa" for row in rows)
    # nt_position is preserved for traceability back to the NT layer.
    assert rows[0]["nt_position"] == 4


def test_cds_window_excludes_outside_mutations():
    # NT mutation at codon 1 (pos 1-3), but CDS starts at NT 4 -> codon 1 of CDS
    # is the second triplet.
    ref = "ATG" "CAT" "TGG"  # M H W
    qry = "ATA" "CAT" "TAT"  # I H Y  -- only the I at codon 1 (NT 1-3) and Y at codon 3
    # With cds_start=4, codon 1 inside CDS is HIS at AA 1, codon 2 is TGG->TAT.
    muts = translate_pair(ref, qry, "sample", cds_start_1based=4)
    subs = [m for m in muts if m.mutation_type == "substitution"]
    # Only the W->Y substitution should appear.
    assert [m.mutation for m in subs] == ["W2Y"]


def test_unequal_lengths_rejected():
    with pytest.raises(ValueError, match="equal length"):
        translate_pair("ATG", "ATGA", "sample")
