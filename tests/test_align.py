"""Tests for seroverse.core.align (SubstitutionMatrix, PairwiseAligner) and consensus seed diagonals."""

import numpy as np
import pytest

from seroverse.core.align.chaining import Chain, Chainer, Chains
from seroverse.core.align.pairwise import PairwiseAligner, PairwiseAlignment, PairwiseAlignments
from seroverse.core.align.score import SubstitutionMatrix
from seroverse.core.align.seeds import Seed, Seeds
from seroverse.core.seq.alphabet import Alphabet
from seroverse.core.seq.clustering import Clusterer
from seroverse.core.seq.kmers import FracMinHashIndex, RandstrobeIndex
from seroverse.core.seq.seq import Sequences


def test_substitution_matrix_singletons_and_lut() -> None:
    """Verify SubstitutionMatrix singleton caching, case-insensitivity, aliases, and bitscores."""
    b62 = SubstitutionMatrix.BLOSUM62
    assert b62.alphabet is Alphabet.AMINO
    assert not b62.lut.flags.writeable
    assert repr(b62) == "SubstitutionMatrix.BLOSUM62"

    # Case-insensitive lookup for W-W (11 in BLOSUM62) and A-R (-1 in BLOSUM62)
    assert int(b62.lut[ord("W"), ord("W")]) == 11
    assert int(b62.lut[ord("w"), ord("W")]) == 11
    assert int(b62.lut[ord("W"), ord("w")]) == 11
    assert int(b62.lut[ord("A"), ord("R")]) == -1
    assert int(b62.lut[ord("a"), ord("r")]) == -1

    # DNA singleton and U->T alias propagation
    dna = SubstitutionMatrix.DNA
    assert dna.alphabet is Alphabet.DNA
    assert int(dna.lut[ord("A"), ord("A")]) == 2
    assert int(dna.lut[ord("A"), ord("C")]) == -3
    assert int(dna.lut[ord("T"), ord("T")]) == 2
    assert int(dna.lut[ord("U"), ord("T")]) == 2
    assert int(dna.lut[ord("u"), ord("t")]) == 2
    assert int(dna.lut[ord("N"), ord("A")]) == -1

    # Singleton identity on repeated construction
    dna_again = SubstitutionMatrix.from_match_mismatch(Alphabet.DNA, match=2, mismatch=-3)
    assert dna_again is dna

    # Mapping constructor
    custom = SubstitutionMatrix(
        Alphabet.DNA,
        {(b"A", b"A"): 5, (b"C", b"C"): 5, (b"G", b"G"): 5, (b"T", b"T"): 5},
        default_score=-4,
    )
    assert int(custom.lut[ord("A"), ord("A")]) == 5
    assert int(custom.lut[ord("A"), ord("G")]) == -4

    # Bitscore scalar and array
    assert b62.bitscore(0) == 0.0
    assert b62.bitscore(-5) == 0.0
    bs_scalar = b62.bitscore(100)
    assert bs_scalar > 40.0
    bs_arr = b62.bitscore(np.array([0, 100], dtype=np.int32))
    assert bs_arr.dtype == np.float32
    assert np.isclose(bs_arr[0], 0.0)
    assert np.isclose(bs_arr[1], bs_scalar, atol=1e-4)

    with pytest.raises(ValueError, match="Score matrix shape"):
        SubstitutionMatrix(Alphabet.DNA, np.zeros((2, 2), dtype=np.int8))


def test_pairwise_aligner_rolling_gotoh_and_dna() -> None:
    """Verify rolling-row + bit-packed uint8 traceback Gotoh alignment for protein and DNA."""
    q_prot = Sequences.from_bytes([b"MKTLLILAVVAAALA", b"ACDEFGHIKLMNPQRSTVWY"])
    t_prot = Sequences.from_bytes([b"MKTLLVLAVVAAALA", b"ACDEFGGGIKLMNPQRSTVWY"])

    aligner = PairwiseAligner()
    alns = aligner(q_prot, t_prot)
    assert len(alns) == 2

    a0 = alns[0]
    assert isinstance(a0, PairwiseAlignment)
    assert a0.matches == 14
    assert a0.mismatches == 1
    assert a0.gaps == 0
    assert a0.q_start == 0
    assert a0.q_end == 15
    assert a0.t_start == 0
    assert a0.t_end == 15
    assert np.isclose(a0.pident, (14.0 / 15.0) * 100.0)
    assert a0.bitscore() > 20.0

    # Alignment with 1 insertion/deletion gap
    a1 = alns[1]
    assert a1.gaps >= 1
    assert a1.q_start == 0
    assert a1.q_end == 20
    assert a1.t_start == 0
    assert a1.t_end == 21

    # DNA alignment with SubstitutionMatrix.DNA
    dna_aligner = PairwiseAligner(matrix=SubstitutionMatrix.DNA, gap_open=5, gap_extend=2, k=10)
    q_dna = Sequences.from_bytes([b"ACGTACGTACGTACGT"])
    t_dna = Sequences.from_bytes([b"ACGTACGTTACGTACGT"])
    dna_alns = dna_aligner(q_dna, t_dna)
    assert len(dna_alns) == 1
    assert dna_alns[0].matches == 16
    assert dna_alns[0].gaps == 1
    assert np.isclose(
        dna_alns[0].bitscore(SubstitutionMatrix.DNA), float(dna_alns.bitscores(SubstitutionMatrix.DNA)[0])
    )

    # Concat and slicing
    combined = PairwiseAlignments.concat([alns[:1], alns[1:]])
    assert len(combined) == 2
    assert np.array_equal(combined.scores, alns.scores)


def test_pairwise_aligner_prefilter_and_search() -> None:
    """Verify MMseqs2-style 4-mer CSR + ungapped diagonal prefiltering and two-stage search."""
    s0 = b"MKTLLILAVVAAALAGSGVASAQITGPVVAAK"
    s1 = b"MKTLLVLAVVAAALAGSGVASAQITGPVVAAK"  # Close homolog of s0
    s2 = b"WYWYWYWYWYWYWYWYWYWYWYWYWYWYWYWY"  # Unrelated sequence
    seqs = Sequences.from_bytes([s0, s1, s2])

    aligner = PairwiseAligner()
    # Self-search upper triangle
    seeds = aligner.prefilter(seqs, min_ungapped_score=35)
    assert len(seeds) == 1
    assert int(seeds[0].query_index) == 0
    assert int(seeds[0].target_index) == 1
    assert int(seeds[0].offset) == 0
    assert int(seeds[0].score) >= 35

    # Bipartite search (queries vs targets) with shifted diagonal
    q_seqs = Sequences.from_bytes([b"GGGGGG" + s0])
    t_seqs = Sequences.from_bytes([s1, s2])
    b_seeds, b_alns = aligner.search(q_seqs, t_seqs, min_ungapped_score=35)
    assert len(b_seeds) == 1
    assert int(b_seeds[0].query_index) == 0
    assert int(b_seeds[0].target_index) == 0
    assert int(b_seeds[0].offset) == 6
    assert len(b_alns) == 1
    assert b_alns[0].pident > 90.0


def test_consensus_diagonal_fixes_first_hit_collision() -> None:
    """Verify modal diagonal voting + ungapped rescoring resolves off-diagonal first-hit collisions."""
    # Construct a 120bp homologous core shared at query[10:130] and target[0:120] (true diagonal = +10),
    # but place a single shared 10-mer repeat at query[0:10] and target[140:150] (spurious first-hit diagonal = -140).
    repeat_10 = b"GCATGCATGC"
    rng = np.random.default_rng(42)
    core = bytes(np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=120)])
    filler_t = bytes(np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=20)])

    query_seq = repeat_10 + core  # length 130: repeat at 0..10, core at 10..130
    target_seq = core + filler_t + repeat_10  # length 150: core at 0..120, repeat at 140..150

    q_seqs = Sequences.from_bytes([query_seq])
    t_seqs = Sequences.from_bytes([target_seq])

    # Use scaled=1 so all 10-mers are indexed; the very first query k-mer at pos 0 hits target pos 140 (d = -140),
    # while ~110 subsequent k-mers hit along the true diagonal d = +10.
    t_idx = FracMinHashIndex.build(t_seqs, k=10, scaled=1, canonical=False, sort_by_hash=True)
    q_idx = FracMinHashIndex.build(q_seqs, k=10, scaled=1, canonical=False, sort_by_hash=False)

    hits = t_idx.hits(q_idx, min_score=2)
    assert len(hits) == 1
    # Modal diagonal voting must select +10 instead of the first hit at -140!
    assert int(hits[0].offset) == 10

    top = t_idx.top_hits(q_idx, min_score=2)
    assert len(top) == 1
    assert int(top[0].offset) == 10

    # Banded alignment using the seed recovers the full 120bp alignment
    dna_aligner = PairwiseAligner(matrix=SubstitutionMatrix.DNA, k=15)
    alns = dna_aligner.align_seeds(q_seqs, t_seqs, hits)
    assert alns[0].matches >= 120
    assert alns[0].pident >= 99.0

    # Test Seeds.rescore_diagonals when an offset is perturbed by a small indel shift
    perturbed = Seeds(
        hits.query_indices,
        hits.target_indices,
        hits.scores,
        np.array([22], dtype=np.int32),
    )
    refined = perturbed.rescore_diagonals(q_seqs, t_seqs, matrix=SubstitutionMatrix.DNA, window=16)
    assert int(refined[0].offset) == 10

    # Also verify RandstrobeIndex hits with query_seqs and target_seqs
    prot_a = b"MKTLLILAVVAAALAGSGVASAQITGPVVAAK" * 2
    prot_b = b"AAAA" + prot_a
    p_seqs = Sequences.from_bytes([prot_a, prot_b])
    r_target = RandstrobeIndex.build(p_seqs, k=5, s=2, sort_by_hash=True)
    r_query = RandstrobeIndex.build(p_seqs, k=5, s=2, sort_by_hash=False)
    r_hits = r_target.hits(r_query, min_score=2, query_seqs=p_seqs, target_seqs=p_seqs)
    pair_01 = r_hits.filter((r_hits.query_indices == 1) & (r_hits.target_indices == 0))
    assert len(pair_01) == 1
    assert int(pair_01[0].offset) == 4


def test_clusterer_with_align_backend() -> None:
    """Verify Clusterer modes work seamlessly with the seroverse.core.align backend."""
    base_a = b"MKTLLILAVVAAALAGSGVASAQITGPVVAAK" * 3
    homolog_a = b"MKTLLVLAVVAAALAGSGVASAQITGPVVAAK" * 3
    base_b = b"WYWYWYWYWYWYWYWYWYWYWYWYWYWYWYWY" * 3
    seqs = Sequences.from_bytes([base_a, homolog_a, base_b])

    for mode in ("cdhit", "mcl", "greedy_cover", "upgma", "connected_components"):
        clusterer = Clusterer(min_identity=80.0, mode=mode)
        clusters = clusterer.cluster(seqs)
        assert clusters.n_clusters == 2
        assert clusters.n_seqs == 3
        c_ids = clusters.cluster_ids
        assert c_ids[0] == c_ids[1]
        assert c_ids[0] != c_ids[2]


def test_unified_seeds_and_unaggregated_hits() -> None:
    """Verify Seed/Seeds anchor enrichment (q_pos, t_pos, spans) and hits(aggregate=False)."""
    rng = np.random.default_rng(7)
    domain_a = bytes(np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=60)])
    indel_gap = b"T" * 18
    domain_b = bytes(np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=60)])

    q_seq = domain_a + indel_gap + domain_b
    t_seq = domain_a + domain_b

    q_seqs = Sequences.from_bytes([q_seq])
    t_seqs = Sequences.from_bytes([t_seq])

    t_idx = FracMinHashIndex.build(t_seqs, k=11, scaled=1, canonical=False, sort_by_hash=True)
    q_idx = FracMinHashIndex.build(q_seqs, k=11, scaled=1, canonical=False, sort_by_hash=False)

    agg_hits = t_idx.hits(q_idx, min_score=2, aggregate=True)
    assert len(agg_hits) == 1
    s0 = agg_hits[0]
    assert isinstance(s0, Seed)
    assert s0.span == 11
    assert s0.q_pos - s0.t_pos == s0.offset

    raw_hits = t_idx.hits(q_idx, min_score=2, aggregate=False)
    assert len(raw_hits) >= 80
    assert np.all(raw_hits.spans == 11)
    assert np.array_equal(raw_hits.q_pos - raw_hits.t_pos, raw_hits.offsets)
    # Verify sorted by (query_index, target_index, t_pos, q_pos)
    assert np.all(np.diff(raw_hits.t_pos) >= 0)
    # Both diagonals (0 for domain_a and +18 for domain_b) are present in raw_hits
    assert np.any(raw_hits.offsets == 0)
    assert np.any(raw_hits.offsets == 18)


def test_pairwise_aligner_block_and_wfa_modes() -> None:
    """Verify adaptive block-shifting Gotoh and WFA2 recover alignments across large indels."""
    rng = np.random.default_rng(123)
    left_arm = bytes(np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=70)])
    right_arm = bytes(np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=70)])
    large_indel = b"A" * 30  # 30bp indel exceeds static k=8 bandwidth

    q_seq = left_arm + large_indel + right_arm
    t_seq = left_arm + right_arm
    q_seqs = Sequences.from_bytes([q_seq])
    t_seqs = Sequences.from_bytes([t_seq])

    seed_0 = Seeds(
        np.array([0], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        np.array([10], dtype=np.uint32),
        np.array([0], dtype=np.int32),
    )

    # Static banded aligner with narrow k=8 cannot cross a 30bp indel when seeded at offset 0
    static_aligner = PairwiseAligner(matrix=SubstitutionMatrix.DNA, gap_open=4, gap_extend=1, k=8, mode="banded")
    static_aln = static_aligner(q_seqs, t_seqs, seeds=seed_0)[0]

    # Adaptive block aligner dynamically shifts its window across the 30bp indel and aligns both 70bp arms
    block_aligner = PairwiseAligner(
        matrix=SubstitutionMatrix.DNA,
        gap_open=4,
        gap_extend=1,
        k=8,
        mode="block",
        block_size=32,
        x_drop=60,
    )
    block_aln = block_aligner(q_seqs, t_seqs, seeds=seed_0)[0]
    assert block_aln.score > static_aln.score
    assert block_aln.matches >= 138
    assert block_aln.gaps == 30
    assert block_aln.q_start == 0
    assert block_aln.q_end == 170
    assert block_aln.t_start == 0
    assert block_aln.t_end == 140

    # WFA2 aligner also recovers the full alignment across the 30bp indel
    wfa_aligner = PairwiseAligner(
        matrix=SubstitutionMatrix.DNA,
        gap_open=4,
        gap_extend=1,
        k=8,
        mode="wfa",
        block_size=32,
        x_drop=60,
    )
    wfa_aln = wfa_aligner(q_seqs, t_seqs)[0]
    assert wfa_aln.matches >= 138
    assert wfa_aln.gaps == 30
    assert wfa_aln.q_start == 0
    assert wfa_aln.q_end == 170
    assert wfa_aln.t_start == 0
    assert wfa_aln.t_end == 140
    assert (wfa_aln.q_end - wfa_aln.q_start) + (wfa_aln.t_end - wfa_aln.t_start) == (
        2 * (wfa_aln.matches + wfa_aln.mismatches) + wfa_aln.gaps
    )


def test_chainer_and_piecewise_chain_alignment() -> None:
    """Verify minimap2-style concave-gap collinear chaining, RaggedArrayContainer Chains, and piecewise alignment."""
    rng = np.random.default_rng(99)
    dom1 = bytes(np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=80)])
    dom2 = bytes(np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=80)])
    dom3 = bytes(np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=80)])
    spacer_q = b"C" * 25
    spacer_t = b"G" * 10

    # Query has dom1 + 25bp insertion + dom2 + dom3
    q_seq = dom1 + spacer_q + dom2 + spacer_t + dom3
    # Target 0 has full homolog (dom1 + dom2 + dom3); Target 1 has only a short paralogous fragment of dom1
    t0_seq = dom1 + dom2 + dom3
    t1_seq = b"AAAA" + dom1[:45] + b"TTTT"

    q_seqs = Sequences.from_bytes([q_seq])
    t_seqs = Sequences.from_bytes([t0_seq, t1_seq])

    t_idx = FracMinHashIndex.build(t_seqs, k=10, scaled=2, canonical=False, sort_by_hash=True)
    raw_seeds = t_idx.hits(q_seqs, min_score=2, aggregate=False)
    assert len(raw_seeds) > 10

    chainer = Chainer(min_cnt=2, min_score=15)
    chains = chainer(raw_seeds)
    assert isinstance(chains, Chains)
    assert len(chains) >= 2

    # Target 0 should be primary, Target 1 (overlapping dom1 on query with lower score) should be secondary
    prim = chains.primary()
    assert len(prim) == 1
    c0 = prim[0]
    assert isinstance(c0, Chain)
    assert c0.query_index == 0
    assert c0.target_index == 0
    assert c0.is_primary is True
    assert len(c0) >= 10
    assert len(c0.seeds) == len(c0)

    # Test Chains container methods: top_chains, to_seeds, concat, slicing
    top = chains.top_chains()
    assert len(top) == 1
    assert top[0] == c0

    as_seeds = prim.to_seeds()
    assert len(as_seeds) == 1
    assert int(as_seeds[0].target_index) == 0

    re_concat = Chains.concat([chains[:1], chains[1:]])
    assert len(re_concat) == len(chains)
    assert re_concat.total_anchors == chains.total_anchors

    # Piecewise base alignment via Chainer.search (WFA2 inter-anchor gaps + adaptive block X-drop ends)
    dna_aligner = PairwiseAligner(matrix=SubstitutionMatrix.DNA, gap_open=4, gap_extend=1)
    found_chains, alns = chainer.search(q_seqs, t_idx, t_seqs, aligner=dna_aligner, min_seed_score=2)
    assert len(found_chains) == 1
    assert len(alns) == 1
    aln = alns[0]
    assert aln.matches >= 235
    assert aln.q_start == 0
    assert aln.q_end == len(q_seq)
    assert aln.t_start == 0
    assert aln.t_end == len(t0_seq)
    # Exact base conservation identity across piecewise WFA2 + block X-drop alignment
    assert (aln.q_end - aln.q_start) + (aln.t_end - aln.t_start) == (2 * (aln.matches + aln.mismatches) + aln.gaps)


def test_scalar_view_and_batched_container_contracts() -> None:
    """Verify ScalarView, BatchedContainer, and RaggedArrayContainer auto-registration and methods."""
    from dataclasses import dataclass

    from seroverse.core.collections import BatchedContainer, Int32Array, ScalarView
    from seroverse.core.graph.arborescence import Arborescence, Arborescences
    from seroverse.core.interval import Interval, Intervals, Strand

    # ScalarView replace, to_dict, and NumPy-safe __eq__ (including Arborescence with np.ndarray fields)
    arb1 = Arborescence(parents=np.array([-1, 0, 1], dtype=np.int32), score=3.5, slots=np.zeros(3, dtype=np.int32))
    arb2 = arb1.replace(score=3.5)
    arb3 = arb1.replace(parents=np.array([-1, 0, 0], dtype=np.int32))
    assert isinstance(arb1, ScalarView)
    assert arb1 == arb2
    assert arb1 != arb3
    assert set(arb1.to_dict()) == {"parents", "score", "slots", "root"}

    # Flat BatchedContainer from_list, __eq__, filter, concat, empty
    iv_list = [Interval(0, 10, Strand.FORWARD), Interval(20, 35, Strand.REVERSE)]
    ivs = Intervals.from_list(iv_list)
    assert len(ivs) == 2
    assert ivs[0] == iv_list[0]
    assert ivs[-1] == iv_list[1]
    assert ivs == Intervals.concat([ivs[:1], ivs[1:]])
    assert len(ivs.filter(np.array([True, False]))) == 1

    # RaggedArrayContainer from_list and __eq__
    arbs = Arborescences.from_list([arb1, arb3])
    assert len(arbs) == 2
    assert arbs[0] == arb1
    assert arbs[1] == arb3
    assert arbs == Arborescences.concat([arbs[:1], arbs[1:]])

    # Enforce 1D NumPy array field annotations at class definition time
    with pytest.raises(TypeError, match="must be annotated as a 1D NumPy array"):

        @dataclass(frozen=True, slots=True)
        class _InvalidContainer(BatchedContainer[Seed]):
            xs: Int32Array
            bad_scalar: int
