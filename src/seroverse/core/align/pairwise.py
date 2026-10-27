r"""Pairwise sequence alignment algorithms, prefiltering engines, and SoA containers.

This module provides high-performance pairwise sequence alignment capabilities
supporting three alignment engines:
- `"block"` (default): Adaptive block-shifting Smith-Waterman-Gotoh (`block-aligner` style)
  with dynamic per-row window steering and X-drop termination.
- `"wfa"`: Affine Wavefront Alignment (WFA2, Marco-Sola et al.) with greedy exact-match
  extension and adaptive wavefront pruning.
- `"banded"`: Classic static-band Smith-Waterman-Gotoh with rolling-row bit-packed traceback.

It also provides MMseqs2-style reduced-alphabet 4-mer CSR + ungapped diagonal prefiltering
and piecewise anchor-guided chain alignment.

Key Classes:
    - [`PairwiseAlignment`][seroverse.core.align.pairwise.PairwiseAlignment]: Immutable container for a single result.
    - [`PairwiseAlignments`][seroverse.core.align.pairwise.PairwiseAlignments]: Batched SoA container for results.
    - [`PairwiseAligner`][seroverse.core.align.pairwise.PairwiseAligner]: Multi-mode pairwise alignment & prefilter engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numba import njit, prange

from seroverse.core.align.score import SubstitutionMatrix, _score_ungapped_diagonal
from seroverse.core.align.seeds import Seeds
from seroverse.core.collections import BatchedContainer, Float32Array, Int32Array, ScalarView
from seroverse.core.seq.alphabet import Alphabet
from seroverse.core.seq.seq import Sequences

if TYPE_CHECKING:
    from seroverse.core.align.chaining import Chains

# Constants ------------------------------------------------------------------------------------------------------------
_INF = np.int32(-1_000_000_000)  # Safe negative infinity for int32 DP arithmetic


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PairwiseAlignment(ScalarView):
    r"""A lightweight, immutable container for the results of a single pairwise sequence alignment.

    This class holds summary statistics and coordinates for an alignment between a single query
    and target sequence. It is typically produced by indexing into a
    [`PairwiseAlignments`][seroverse.core.align.pairwise.PairwiseAlignments] collection.

    Attributes:
        score (int): Final alignment score calculated using the substitution matrix and gap penalties.
        matches (int): Total number of matching residues/bases.
        mismatches (int): Total number of mismatched residues/bases.
        gaps (int): Total number of gap characters (insertions or deletions).
        q_start (int): 0-based start coordinate on query sequence (inclusive).
        q_end (int): 0-based end coordinate on query sequence (exclusive).
        t_start (int): 0-based start coordinate on target sequence (inclusive).
        t_end (int): 0-based end coordinate on target sequence (exclusive).
    """

    score: int
    matches: int
    mismatches: int
    gaps: int
    q_start: int
    q_end: int
    t_start: int
    t_end: int

    @property
    def pident(self) -> float:
        r"""Calculate the percent identity of the aligned region.

        Percent identity is defined as `(matches / (matches + mismatches + gaps)) * 100.0`.

        Returns:
            float: Percent identity value between 0.0 and 100.0, or 0.0 if alignment length is zero.
        """
        total = self.matches + self.mismatches + self.gaps
        return (self.matches / total) * 100.0 if total > 0 else 0.0

    def bitscore(self, matrix: SubstitutionMatrix = SubstitutionMatrix.BLOSUM62) -> float:
        r"""Calculate the Karlin-Altschul bitscore of the alignment using `matrix`.

        Uses the substitution matrix parameters ($\lambda, K$):
        $B = (\lambda S - \ln K) / \ln 2$ for $S > 0$, else $0.0$.
        """
        return matrix.bitscore(self.score)


@dataclass(frozen=True, slots=True)
class PairwiseAlignments(BatchedContainer[PairwiseAlignment, "PairwiseAlignments"]):
    r"""A high-performance SoA container for the results of multiple pairwise alignments.

    This class stores alignment statistics in a Structure-of-Arrays (SoA) layout using 1D NumPy arrays.

    Attributes:
        scores (Int32Array): 1D array of alignment scores.
        matches (Int32Array): 1D array of match counts.
        mismatches (Int32Array): 1D array of mismatch counts.
        gaps (Int32Array): 1D array of gap counts.
        q_starts (Int32Array): 1D array of query start coordinates.
        q_ends (Int32Array): 1D array of query end coordinates.
        t_starts (Int32Array): 1D array of target start coordinates.
        t_ends (Int32Array): 1D array of target end coordinates.
    """

    scores: Int32Array
    matches: Int32Array
    mismatches: Int32Array
    gaps: Int32Array
    q_starts: Int32Array
    q_ends: Int32Array
    t_starts: Int32Array
    t_ends: Int32Array

    @property
    def pidents(self) -> Float32Array:
        r"""Calculate percent identity for all alignments in the batch in a vectorized manner.

        Returns:
            Float32Array: 1D array of percent identity values.
        """
        total = self.matches + self.mismatches + self.gaps
        return np.divide(self.matches * 100.0, total, out=np.zeros(len(self), dtype=np.float32), where=total > 0)

    def bitscores(self, matrix: SubstitutionMatrix = SubstitutionMatrix.BLOSUM62) -> Float32Array:
        r"""Calculate Karlin-Altschul bitscores for all alignments in the batch using `matrix`.

        Args:
            matrix (SubstitutionMatrix): Substitution matrix used for $(\lambda, K)$ parameters.
                Defaults to `SubstitutionMatrix.BLOSUM62`.

        Returns:
            Float32Array: 1D array of bitscores ($\ge 0.0$).
        """
        return matrix.bitscore(self.scores)


@dataclass(frozen=True, slots=True)
class PairwiseAligner:
    r"""A high-performance, batched pairwise sequence aligner and MMseqs2-style prefilter engine.

    Supports three execution modes:
    - `"block"` (default): Adaptive block-shifting Smith-Waterman-Gotoh (`block-aligner` style)
      that dynamically shifts the active diagonal block per row to track arbitrary indels without
      falling out of a rigid static band, with `x_drop` early termination.
    - `"wfa"`: Affine Wavefront Alignment (WFA2, Marco-Sola et al.) with greedy exact-match slides
      and adaptive wavefront pruning in $O(N + s^2)$ time.
    - `"banded"`: Classic static-band Smith-Waterman-Gotoh within a fixed `2*k + 1` diagonal band.

    Attributes:
        matrix (SubstitutionMatrix): Substitution matrix for scoring. Defaults to `SubstitutionMatrix.BLOSUM62`.
        gap_open (int): Penalty for opening a new gap. Defaults to 11.
        gap_extend (int): Penalty for extending an existing gap. Defaults to 1.
        k (int): Bandwidth parameter (`2*k+1` diagonals). Defaults to 20.
        mode (Literal["block", "banded", "wfa"]): Alignment algorithm mode. Defaults to `"block"`.
        block_size (int): Half-block width parameter for `"block"` and `"wfa"` pruning. Defaults to 32.
        x_drop (int): X-drop score threshold for early termination and block shifting. Defaults to 50.
    """

    matrix: SubstitutionMatrix = SubstitutionMatrix.BLOSUM62
    gap_open: int = 11
    gap_extend: int = 1
    k: int = 20
    mode: Literal["block", "banded", "wfa"] = "block"
    block_size: int = 32
    x_drop: int = 50

    def __call__(self, queries: Sequences, targets: Sequences, seeds: Seeds | None = None) -> PairwiseAlignments:
        r"""Perform pairwise alignment of query and target sequences.

        Args:
            queries (Sequences): Collection of query sequences.
            targets (Sequences): Collection of target sequences (must match query batch size).
            seeds (Seeds | None): Optional alignment seeds guiding diagonal alignment.

        Returns:
            PairwiseAlignments: Alignment scores, statistics, and coordinates for each sequence pair.

        Raises:
            ValueError: If query and target batches have different numbers of sequences or `mode` is invalid.
        """
        if len(queries.offsets) != len(targets.offsets):
            raise ValueError("Query and target batches must have the same number of sequences.")

        n = len(queries.offsets)
        if n == 0:
            return PairwiseAlignments.empty()

        if seeds is not None:
            is_seeded = True
            offsets_arr = seeds.offsets
        else:
            is_seeded = False
            offsets_arr = np.zeros(n, dtype=np.int32)

        out_scores = np.empty(n, dtype=np.int32)
        out_matches = np.empty(n, dtype=np.int32)
        out_mismatches = np.empty(n, dtype=np.int32)
        out_gaps = np.empty(n, dtype=np.int32)
        out_q_starts = np.empty(n, dtype=np.int32)
        out_q_ends = np.empty(n, dtype=np.int32)
        out_t_starts = np.empty(n, dtype=np.int32)
        out_t_ends = np.empty(n, dtype=np.int32)

        if self.mode == "banded":
            _batched_banded_gotoh(
                queries.seqs,
                queries.offsets,
                queries.lengths,
                targets.seqs,
                targets.offsets,
                targets.lengths,
                self.matrix.lut,
                self.k,
                self.gap_open,
                self.gap_extend,
                is_seeded,
                offsets_arr,
                out_scores,
                out_matches,
                out_mismatches,
                out_gaps,
                out_q_starts,
                out_q_ends,
                out_t_starts,
                out_t_ends,
            )
        elif self.mode == "block":
            _batched_block_gotoh(
                queries.seqs,
                queries.offsets,
                queries.lengths,
                targets.seqs,
                targets.offsets,
                targets.lengths,
                self.matrix.lut,
                self.k,
                self.block_size,
                self.gap_open,
                self.gap_extend,
                self.x_drop,
                is_seeded,
                offsets_arr,
                out_scores,
                out_matches,
                out_mismatches,
                out_gaps,
                out_q_starts,
                out_q_ends,
                out_t_starts,
                out_t_ends,
            )
        elif self.mode == "wfa":
            _batched_wfa(
                queries.seqs,
                queries.offsets,
                queries.lengths,
                targets.seqs,
                targets.offsets,
                targets.lengths,
                self.matrix.lut,
                self.k,
                self.block_size,
                self.gap_open,
                self.gap_extend,
                self.x_drop,
                is_seeded,
                offsets_arr,
                out_scores,
                out_matches,
                out_mismatches,
                out_gaps,
                out_q_starts,
                out_q_ends,
                out_t_starts,
                out_t_ends,
            )
        else:
            raise ValueError(f"Unsupported PairwiseAligner mode: {self.mode!r}. Expected 'block', 'banded', or 'wfa'.")

        return PairwiseAlignments(
            out_scores,
            out_matches,
            out_mismatches,
            out_gaps,
            out_q_starts,
            out_q_ends,
            out_t_starts,
            out_t_ends,
        )

    def align_seeds(self, queries: Sequences, targets: Sequences, seeds: Seeds) -> PairwiseAlignments:
        r"""Convenience method to extract and align specific sequence pairs mapped by seeds.

        Args:
            queries (Sequences): Full collection of query sequences.
            targets (Sequences): Full collection of target sequences.
            seeds (Seeds): Seed collection mapping specific query sequences to target sequences.

        Returns:
            PairwiseAlignments: Alignment results parallel to the provided seeds.
        """
        if len(seeds) == 0:
            return PairwiseAlignments.empty()
        paired_queries, paired_targets = seeds.extract_sequences(queries, targets)
        return self(paired_queries, paired_targets, seeds)

    def align_chains(self, queries: Sequences, targets: Sequences, chains: Chains) -> PairwiseAlignments:
        r"""Perform piecewise anchor-guided base alignment for a batch of collinear [`Chains`][seroverse.core.align.chaining.Chains].

        Aligns inter-anchor gaps using pinned Affine Wavefront Alignment (WFA2) and extends
        left and right chain ends using adaptive block X-drop Smith-Waterman-Gotoh.

        Args:
            queries (Sequences): Full collection of query sequences.
            targets (Sequences): Full collection of target sequences.
            chains (Chains): Collinear anchor chains mapping query sequences to target sequences.

        Returns:
            PairwiseAlignments: Base-level alignment results parallel to `chains`.
        """
        n_chains = len(chains)
        if n_chains == 0:
            return PairwiseAlignments.empty()

        out_scores = np.empty(n_chains, dtype=np.int32)
        out_matches = np.empty(n_chains, dtype=np.int32)
        out_mismatches = np.empty(n_chains, dtype=np.int32)
        out_gaps = np.empty(n_chains, dtype=np.int32)
        out_q_starts = np.empty(n_chains, dtype=np.int32)
        out_q_ends = np.empty(n_chains, dtype=np.int32)
        out_t_starts = np.empty(n_chains, dtype=np.int32)
        out_t_ends = np.empty(n_chains, dtype=np.int32)

        _align_chains_kernel(
            queries.seqs,
            queries.offsets,
            queries.lengths,
            targets.seqs,
            targets.offsets,
            targets.lengths,
            chains.query_indices,
            chains.target_indices,
            chains.offsets,
            chains.lengths,
            chains.anchor_q_pos,
            chains.anchor_t_pos,
            chains.anchor_spans,
            self.matrix.lut,
            max(self.k, self.block_size),
            self.gap_open,
            self.gap_extend,
            self.x_drop,
            out_scores,
            out_matches,
            out_mismatches,
            out_gaps,
            out_q_starts,
            out_q_ends,
            out_t_starts,
            out_t_ends,
        )

        return PairwiseAlignments(
            out_scores,
            out_matches,
            out_mismatches,
            out_gaps,
            out_q_starts,
            out_q_ends,
            out_t_starts,
            out_t_ends,
        )

    def prefilter(
        self,
        queries: Sequences,
        targets: Sequences | None = None,
        *,
        min_ungapped_score: int = 42,
        min_coverage: float = 0.0,
        upper_triangle: bool | None = None,
    ) -> Seeds:
        r"""Run MMseqs2-style 4-mer CSR + ungapped diagonal prefiltering to discover candidate homologous pairs.

        Builds a direct $12^4 = 20,736$-bucket CSR inverted index of 4-mers (`Alphabet.MMSEQS12` for
        proteins or `Alphabet.DNA` for nucleotides) on `targets` (or `queries` for self-search) and
        evaluates ungapped diagonal scores using `self.matrix` on shared 4-mer diagonals.

        Args:
            queries (Sequences): Query sequences.
            targets (Sequences | None): Target sequences. If `None`, performs self-search on `queries`.
            min_ungapped_score (int): Minimum local ungapped diagonal score to retain a pair. Defaults to `42`.
            min_coverage (float): Minimum length ratio $\min(L_q, L_t) / \max(L_q, L_t)$. Defaults to `0.0`.
            upper_triangle (bool | None): If `True`, restricts output to pairs with `q_idx < t_idx`.
                Defaults to `True` when `targets is None` and `False` otherwise.

        Returns:
            Seeds: Candidate [`Seeds`][seroverse.core.align.seeds.Seeds] containing `(query_indices, target_indices, scores, offsets, q_pos, t_pos, spans)`.
        """
        is_self = targets is None or targets is queries
        t_seqs = queries if targets is None else targets
        if len(queries) == 0 or len(t_seqs) == 0:
            return Seeds.empty()

        use_upper = is_self if upper_triangle is None else bool(upper_triangle)
        kmer_lut = Alphabet.DNA.kmer_lut if self.matrix.alphabet.supports_canonical else Alphabet.MMSEQS12.kmer_lut

        out_q, out_t, out_scores, out_diags, out_qp, out_tp = _mmseqs12_k4_ungapped_prefilter(
            queries.seqs,
            queries.offsets,
            queries.lengths,
            t_seqs.seqs,
            t_seqs.offsets,
            t_seqs.lengths,
            float(min_coverage),
            int(min_ungapped_score),
            kmer_lut,
            self.matrix.lut,
            use_upper,
        )
        out_sp = np.full(len(out_q), 4, dtype=np.uint16)
        return Seeds(out_q, out_t, out_scores, out_diags, out_qp, out_tp, out_sp)

    def search(
        self,
        queries: Sequences,
        targets: Sequences | None = None,
        *,
        min_ungapped_score: int = 42,
        min_coverage: float = 0.0,
        upper_triangle: bool | None = None,
        banded: bool = True,
    ) -> tuple[Seeds, PairwiseAlignments]:
        r"""Execute a two-stage MMseqs2-style prefilter followed by Smith-Waterman-Gotoh / WFA alignment.

        Args:
            queries (Sequences): Query sequences.
            targets (Sequences | None): Target sequences. If `None`, performs self-search on `queries`.
            min_ungapped_score (int): Minimum ungapped diagonal score threshold for Stage 1. Defaults to `42`.
            min_coverage (float): Minimum sequence length ratio filter. Defaults to `0.0`.
            upper_triangle (bool | None): Restrict to `q_idx < t_idx` (defaults to `True` when `targets is None`).
            banded (bool): If `True` (default), seeds the aligner with the Stage 1 diagonal offsets.
                If `False`, runs unseeded alignment on the prefiltered pairs.

        Returns:
            tuple[Seeds, PairwiseAlignments]: Parallel `(seeds, alignments)` for all candidate pairs passing Stage 1.
        """
        t_seqs = queries if targets is None else targets
        seeds = self.prefilter(
            queries,
            targets,
            min_ungapped_score=min_ungapped_score,
            min_coverage=min_coverage,
            upper_triangle=upper_triangle,
        )
        if len(seeds) == 0:
            return seeds, PairwiseAlignments.empty()
        if banded:
            alignments = self.align_seeds(queries, t_seqs, seeds)
        else:
            paired_q, paired_t = seeds.extract_sequences(queries, t_seqs)
            alignments = self(paired_q, paired_t, seeds=None)
        return seeds, alignments


# Kernels --------------------------------------------------------------------------------------------------------------
@njit(nogil=True, cache=True)
def _build_mmseqs12_k4_csr(
    seqs_data: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    lut: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Build a direct $12^4 = 20,736$-bucket CSR inverted index of 4-mers (`Alphabet.MMSEQS12` or `Alphabet.DNA`)."""
    n_buckets = 20736
    counts = np.zeros(n_buckets + 1, dtype=np.int32)
    n_seqs = len(lengths)
    for u in range(n_seqs):
        seq_len = lengths[u]
        s = offsets[u]
        for i in range(seq_len - 3):
            c0 = lut[seqs_data[s + i]]
            c1 = lut[seqs_data[s + i + 1]]
            c2 = lut[seqs_data[s + i + 2]]
            c3 = lut[seqs_data[s + i + 3]]
            if c0 < 12 and c1 < 12 and c2 < 12 and c3 < 12:
                counts[((c0 * 12 + c1) * 12 + c2) * 12 + c3 + 1] += 1
    indptr = np.zeros(n_buckets + 1, dtype=np.int32)
    for b in range(n_buckets):
        indptr[b + 1] = indptr[b] + counts[b + 1]
    post_seq = np.empty(indptr[n_buckets], dtype=np.int32)
    post_pos = np.empty(indptr[n_buckets], dtype=np.int16)
    cur = indptr[:-1].copy()
    for u in range(n_seqs):
        seq_len = lengths[u]
        s = offsets[u]
        for i in range(seq_len - 3):
            c0 = lut[seqs_data[s + i]]
            c1 = lut[seqs_data[s + i + 1]]
            c2 = lut[seqs_data[s + i + 2]]
            c3 = lut[seqs_data[s + i + 3]]
            if c0 < 12 and c1 < 12 and c2 < 12 and c3 < 12:
                code = ((c0 * 12 + c1) * 12 + c2) * 12 + c3
                idx = cur[code]
                post_seq[idx] = u
                post_pos[idx] = i
                cur[code] = idx + 1
    return indptr, post_seq, post_pos


@njit(parallel=True, nogil=True, cache=True)
def _mmseqs12_k4_ungapped_prefilter(
    q_data: np.ndarray,
    q_offsets: np.ndarray,
    q_lengths: np.ndarray,
    t_data: np.ndarray,
    t_offsets: np.ndarray,
    t_lengths: np.ndarray,
    min_cov: float,
    min_ungapped_score: int,
    lut: np.ndarray,
    matrix: np.ndarray,
    upper_triangle: bool = True,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
]:
    r"""MMseqs2-style reduced-alphabet 4-mer CSR + ungapped diagonal prefilter."""
    indptr, post_seq, post_pos = _build_mmseqs12_k4_csr(t_data, t_offsets, t_lengths, lut)
    n_queries = len(q_lengths)
    n_targets = len(t_lengths)
    row_counts = np.zeros(n_queries, dtype=np.int32)

    for u in prange(n_queries):  # type: ignore[not-iterable]
        l1 = q_lengths[u]
        s1 = q_offsets[u]
        best_sc = np.zeros(n_targets, dtype=np.int16)
        last_diag = np.full(n_targets, 32767, dtype=np.int16)
        for i in range(l1 - 3):
            c0 = lut[q_data[s1 + i]]
            c1 = lut[q_data[s1 + i + 1]]
            c2 = lut[q_data[s1 + i + 2]]
            c3 = lut[q_data[s1 + i + 3]]
            if c0 < 12 and c1 < 12 and c2 < 12 and c3 < 12:
                code = ((c0 * 12 + c1) * 12 + c2) * 12 + c3
                for idx in range(indptr[code], indptr[code + 1]):
                    v = post_seq[idx]
                    if upper_triangle and v <= u:
                        continue
                    l2 = t_lengths[v]
                    if min_cov > 0.0 and min(l1, l2) < min_cov * max(l1, l2):
                        continue
                    d = i - int(post_pos[idx])
                    if d == last_diag[v] or best_sc[v] >= min_ungapped_score:
                        continue
                    last_diag[v] = d
                    min_span = min(12, l1, l2)
                    sc = _score_ungapped_diagonal(q_data, s1, l1, t_data, t_offsets[v], l2, d, matrix, min_span)
                    if sc > best_sc[v]:
                        best_sc[v] = min(sc, 32760)
        cnt = 0
        v_start = (u + 1) if upper_triangle else 0
        for v in range(v_start, n_targets):
            if best_sc[v] >= min_ungapped_score:
                cnt += 1
        row_counts[u] = cnt

    out_offsets = np.zeros(n_queries + 1, dtype=np.int32)
    for u in range(n_queries):
        out_offsets[u + 1] = out_offsets[u] + row_counts[u]
    total_hits = out_offsets[n_queries]
    out_q = np.empty(total_hits, dtype=np.uint32)
    out_t = np.empty(total_hits, dtype=np.uint32)
    out_scores = np.empty(total_hits, dtype=np.uint32)
    out_diags = np.empty(total_hits, dtype=np.int32)
    out_qp = np.empty(total_hits, dtype=np.int32)
    out_tp = np.empty(total_hits, dtype=np.int32)

    for u in prange(n_queries):  # type: ignore[not-iterable]
        if row_counts[u] == 0:
            continue
        l1 = q_lengths[u]
        s1 = q_offsets[u]
        best_sc = np.zeros(n_targets, dtype=np.int16)
        best_diag = np.zeros(n_targets, dtype=np.int32)
        best_q = np.zeros(n_targets, dtype=np.int32)
        best_t = np.zeros(n_targets, dtype=np.int32)
        last_diag = np.full(n_targets, 32767, dtype=np.int16)
        for i in range(l1 - 3):
            c0 = lut[q_data[s1 + i]]
            c1 = lut[q_data[s1 + i + 1]]
            c2 = lut[q_data[s1 + i + 2]]
            c3 = lut[q_data[s1 + i + 3]]
            if c0 < 12 and c1 < 12 and c2 < 12 and c3 < 12:
                code = ((c0 * 12 + c1) * 12 + c2) * 12 + c3
                for idx in range(indptr[code], indptr[code + 1]):
                    v = post_seq[idx]
                    if upper_triangle and v <= u:
                        continue
                    l2 = t_lengths[v]
                    if min_cov > 0.0 and min(l1, l2) < min_cov * max(l1, l2):
                        continue
                    t_pos_val = int(post_pos[idx])
                    d = i - t_pos_val
                    if d == last_diag[v] or best_sc[v] >= min_ungapped_score:
                        continue
                    last_diag[v] = d
                    min_span = min(12, l1, l2)
                    sc = _score_ungapped_diagonal(q_data, s1, l1, t_data, t_offsets[v], l2, d, matrix, min_span)
                    if sc > best_sc[v]:
                        best_sc[v] = min(sc, 32760)
                        best_diag[v] = d
                        best_q[v] = i
                        best_t[v] = t_pos_val
        w = out_offsets[u]
        v_start = (u + 1) if upper_triangle else 0
        for v in range(v_start, n_targets):
            if best_sc[v] >= min_ungapped_score:
                out_q[w] = np.uint32(u)
                out_t[w] = np.uint32(v)
                out_scores[w] = np.uint32(best_sc[v])
                out_diags[w] = best_diag[v]
                out_qp[w] = best_q[v]
                out_tp[w] = best_t[v]
                w += 1
    return out_q, out_t, out_scores, out_diags, out_qp, out_tp


@njit(cache=True, nogil=True)
def _banded_gotoh_pair(
    seq1: np.ndarray[tuple[int], np.dtype[np.uint8]],
    seq2: np.ndarray[tuple[int], np.dtype[np.uint8]],
    matrix: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    k: int,
    gap_open: int,
    gap_extend: int,
    is_seeded: bool,
    offset: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    r"""Classic static-band Smith-Waterman-Gotoh alignment for a single sequence pair."""
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0 or len2 == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    rows, cols = len1 + 1, len2 + 1
    if is_seeded:
        k_local = k
        diag_offset = offset
    else:
        k_local = max(k, abs(len1 - len2) + 1)
        diag_offset = 0

    band_width = 2 * k_local + 3
    M_prev = np.zeros(band_width, dtype=np.int32)
    M_curr = np.zeros(band_width, dtype=np.int32)
    D_prev = np.full(band_width, _INF, dtype=np.int32)
    D_curr = np.full(band_width, _INF, dtype=np.int32)
    I_curr = np.full(band_width, _INF, dtype=np.int32)
    tb = np.full((rows, band_width), 3, dtype=np.uint8)

    max_score = 0
    max_i = 0
    max_j = 0

    for i in range(1, rows):
        for b_idx in range(band_width):
            M_curr[b_idx] = 0
            D_curr[b_idx] = _INF
            I_curr[b_idx] = _INF

        j_center = i - diag_offset
        start_j = max(1, j_center - k_local)
        end_j = min(cols, j_center + k_local + 1)

        if start_j < cols and end_j > 1:
            start_j_prev = max(0, i - 1 - diag_offset - k_local - 1)
            start_j_curr = max(0, i - diag_offset - k_local - 1)

            for j in range(start_j, end_j):
                jm_top = j - start_j_prev
                d_open = M_prev[jm_top] - gap_open - gap_extend
                d_ext = D_prev[jm_top] - gap_extend

                jm = j - start_j_curr
                if d_open >= d_ext:
                    d_val = d_open
                    tb_d_bit = 0
                else:
                    d_val = d_ext
                    tb_d_bit = 4
                D_curr[jm] = d_val

                jm_left = j - 1 - start_j_curr
                i_open = M_curr[jm_left] - gap_open - gap_extend
                i_ext = I_curr[jm_left] - gap_extend
                if i_open >= i_ext:
                    i_val = i_open
                    tb_i_bit = 0
                else:
                    i_val = i_ext
                    tb_i_bit = 8
                I_curr[jm] = i_val

                jm_top_left = j - 1 - start_j_prev
                m_diag = M_prev[jm_top_left] + int(matrix[seq1[i - 1], seq2[j - 1]])

                best = m_diag
                tb_m_bits = 0
                if d_val > best:
                    best = d_val
                    tb_m_bits = 1
                if i_val > best:
                    best = i_val
                    tb_m_bits = 2

                if best <= 0:
                    M_curr[jm] = 0
                    tb[i, jm] = np.uint8(3 | tb_d_bit | tb_i_bit)
                else:
                    M_curr[jm] = best
                    tb[i, jm] = np.uint8(tb_m_bits | tb_d_bit | tb_i_bit)
                    if best > max_score:
                        max_score = best
                        max_i = i
                        max_j = j

        M_prev, M_curr = M_curr, M_prev
        D_prev, D_curr = D_curr, D_prev

    i, j = max_i, max_j
    matches = mismatches = gaps = state = 0
    end_i, end_j = i, j

    while i > 0 and j > 0:
        start_j_curr = max(0, i - diag_offset - k_local - 1)
        jm = j - start_j_curr
        tb_val = tb[i, jm]

        if state == 0:
            tb_m = tb_val & 3
            if tb_m == 3:
                break
            elif tb_m == 0:
                if seq1[i - 1] == seq2[j - 1]:
                    matches += 1
                else:
                    mismatches += 1
                i -= 1
                j -= 1
            else:
                state = int(tb_m)
        elif state == 1:
            tb_d = (tb_val >> 2) & 1
            gaps += 1
            i -= 1
            if tb_d == 0:
                state = 0
        elif state == 2:
            tb_i = (tb_val >> 3) & 1
            gaps += 1
            j -= 1
            if tb_i == 0:
                state = 0

    return max_score, matches, mismatches, gaps, i, end_i, j, end_j


@njit(cache=True, nogil=True)
def _block_gotoh_pair(
    seq1: np.ndarray[tuple[int], np.dtype[np.uint8]],
    seq2: np.ndarray[tuple[int], np.dtype[np.uint8]],
    matrix: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    k: int,
    block_size: int,
    gap_open: int,
    gap_extend: int,
    x_drop: int,
    is_seeded: bool,
    offset: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    r"""Adaptive block-shifting Smith-Waterman-Gotoh (`block-aligner` style) for a single sequence pair.

    Dynamically shifts the active row window `row_start_j[i]` to track the highest-scoring column
    across arbitrary insertions and deletions, terminating early when the row maximum drops by `x_drop`.
    """
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0 or len2 == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    rows, cols = len1 + 1, len2 + 1
    half_w = max(k, int(block_size * 1.5)) if is_seeded else max(k, int(block_size * 1.5), abs(len1 - len2) + 1)
    active_offset = offset if is_seeded else 0

    band_width = 2 * half_w + 3
    M_prev = np.zeros(band_width, dtype=np.int32)
    M_curr = np.zeros(band_width, dtype=np.int32)
    D_prev = np.full(band_width, _INF, dtype=np.int32)
    D_curr = np.full(band_width, _INF, dtype=np.int32)
    I_curr = np.full(band_width, _INF, dtype=np.int32)

    tb = np.full((rows, band_width), 3, dtype=np.uint8)
    row_start_j = np.zeros(rows, dtype=np.int32)
    row_start_j[0] = max(0, -active_offset - half_w - 1)

    max_score = 0
    max_i = 0
    max_j = 0
    shift_tol = max(4, half_w // 4)

    for i in range(1, rows):
        for b_idx in range(band_width):
            M_curr[b_idx] = 0
            D_curr[b_idx] = _INF
            I_curr[b_idx] = _INF

        j_center = i - active_offset
        start_j = max(1, j_center - half_w)
        end_j = min(cols, j_center + half_w + 1)
        start_j_curr = max(0, j_center - half_w - 1)
        row_start_j[i] = start_j_curr
        start_j_prev = row_start_j[i - 1]

        row_best_score = 0
        row_best_j = j_center

        if start_j < cols and end_j > 1:
            for j in range(start_j, end_j):
                jm = j - start_j_curr
                jm_top = j - start_j_prev
                if 0 <= jm_top < band_width:
                    d_open = M_prev[jm_top] - gap_open - gap_extend
                    d_ext = D_prev[jm_top] - gap_extend
                else:
                    d_open = _INF
                    d_ext = _INF

                if d_open >= d_ext:
                    d_val = d_open
                    tb_d_bit = 0
                else:
                    d_val = d_ext
                    tb_d_bit = 4
                D_curr[jm] = d_val

                jm_left = j - 1 - start_j_curr
                if 0 <= jm_left < band_width:
                    i_open = M_curr[jm_left] - gap_open - gap_extend
                    i_ext = I_curr[jm_left] - gap_extend
                else:
                    i_open = _INF
                    i_ext = _INF

                if i_open >= i_ext:
                    i_val = i_open
                    tb_i_bit = 0
                else:
                    i_val = i_ext
                    tb_i_bit = 8
                I_curr[jm] = i_val

                jm_top_left = j - 1 - start_j_prev
                m_prev_val = M_prev[jm_top_left] if 0 <= jm_top_left < band_width else 0
                m_diag = m_prev_val + int(matrix[seq1[i - 1], seq2[j - 1]])

                best = m_diag
                tb_m_bits = 0
                if d_val > best:
                    best = d_val
                    tb_m_bits = 1
                if i_val > best:
                    best = i_val
                    tb_m_bits = 2

                if best <= 0:
                    M_curr[jm] = 0
                    tb[i, jm] = np.uint8(3 | tb_d_bit | tb_i_bit)
                else:
                    M_curr[jm] = best
                    tb[i, jm] = np.uint8(tb_m_bits | tb_d_bit | tb_i_bit)
                    if best > row_best_score:
                        row_best_score = best
                        row_best_j = j
                    if best > max_score:
                        max_score = best
                        max_i = i
                        max_j = j

        # Adaptive block-shifting: steer active_offset to follow indels once inside homologous region,
        # or return to the seed diagonal if the local alignment score has decayed below x_drop.
        if x_drop > 0 and max_score >= 25 and row_best_score < max_score - x_drop:
            active_offset = offset if is_seeded else 0
        elif row_best_score >= 15:
            best_diag = i - row_best_j
            if abs(best_diag - active_offset) > shift_tol:
                active_offset = best_diag

        M_prev, M_curr = M_curr, M_prev
        D_prev, D_curr = D_curr, D_prev

    i, j = max_i, max_j
    matches = mismatches = gaps = state = 0
    end_i, end_j = i, j

    while i > 0 and j > 0:
        start_j_curr = row_start_j[i]
        jm = j - start_j_curr
        if jm < 0 or jm >= band_width:
            break
        tb_val = tb[i, jm]

        if state == 0:
            tb_m = tb_val & 3
            if tb_m == 3:
                break
            elif tb_m == 0:
                if seq1[i - 1] == seq2[j - 1]:
                    matches += 1
                else:
                    mismatches += 1
                i -= 1
                j -= 1
            else:
                state = int(tb_m)
        elif state == 1:
            tb_d = (tb_val >> 2) & 1
            gaps += 1
            i -= 1
            if tb_d == 0:
                state = 0
        elif state == 2:
            tb_i = (tb_val >> 3) & 1
            gaps += 1
            j -= 1
            if tb_i == 0:
                state = 0

    return int(max_score), matches, mismatches, gaps, i, end_i, j, end_j


@njit(cache=True, nogil=True)
def _block_xdrop_extend(
    seq1: np.ndarray[tuple[int], np.dtype[np.uint8]],
    seq2: np.ndarray[tuple[int], np.dtype[np.uint8]],
    matrix: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    block_size: int,
    gap_open: int,
    gap_extend: int,
    x_drop: int,
    reverse: bool,
) -> tuple[int, int, int, int, int, int]:
    r"""Pinned X-drop adaptive block extension from `(0, 0)` (or backward from `(len1, len2)` when `reverse=True`).

    Returns `(max_score, matches, mismatches, gaps, ext_q_len, ext_t_len)`.
    """
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0 or len2 == 0:
        return 0, 0, 0, 0, 0, 0

    rows, cols = len1 + 1, len2 + 1
    half_w = max(16, block_size)
    band_width = 2 * half_w + 3

    M_prev = np.full(band_width, _INF, dtype=np.int32)
    M_curr = np.full(band_width, _INF, dtype=np.int32)
    D_prev = np.full(band_width, _INF, dtype=np.int32)
    D_curr = np.full(band_width, _INF, dtype=np.int32)
    I_curr = np.full(band_width, _INF, dtype=np.int32)

    tb = np.full((rows, band_width), 3, dtype=np.uint8)
    row_start_j = np.zeros(rows, dtype=np.int32)

    active_offset = 0
    start_j_0 = max(0, -active_offset - half_w - 1)
    row_start_j[0] = start_j_0

    # Initialize row 0 pinned at (0, 0)
    jm0 = 0 - start_j_0
    if 0 <= jm0 < band_width:
        M_prev[jm0] = 0
        tb[0, jm0] = 3
    max_j_init = min(cols, half_w + 1)
    for j in range(1, max_j_init):
        jm = j - start_j_0
        if 0 <= jm < band_width:
            g_sc = -(gap_open + j * gap_extend)
            if g_sc >= -x_drop:
                M_prev[jm] = g_sc
                tb[0, jm] = np.uint8(2 | (8 if j > 1 else 0))

    max_score = 0
    max_i = 0
    max_j = 0
    shift_tol = max(4, half_w // 4)

    for i in range(1, rows):
        for b_idx in range(band_width):
            M_curr[b_idx] = _INF
            D_curr[b_idx] = _INF
            I_curr[b_idx] = _INF

        j_center = i - active_offset
        start_j_curr = max(0, j_center - half_w - 1)
        row_start_j[i] = start_j_curr
        start_j_prev = row_start_j[i - 1]

        # Column j = 0 (pure vertical gap from (0, 0))
        if start_j_curr == 0:
            v_sc = -(gap_open + i * gap_extend)
            if v_sc >= max_score - x_drop:
                D_curr[0] = v_sc
                M_curr[0] = v_sc
                tb[i, 0] = np.uint8(1 | (4 if i > 1 else 0))

        start_j = max(1, j_center - half_w)
        end_j = min(cols, j_center + half_w + 1)
        row_best_score = _INF
        row_best_j = j_center

        c1 = seq1[len1 - i] if reverse else seq1[i - 1]

        if start_j < cols and end_j > 1:
            for j in range(start_j, end_j):
                jm = j - start_j_curr
                jm_top = j - start_j_prev
                if 0 <= jm_top < band_width:
                    d_open = M_prev[jm_top] - gap_open - gap_extend
                    d_ext = D_prev[jm_top] - gap_extend
                else:
                    d_open = _INF
                    d_ext = _INF

                if d_open >= d_ext:
                    d_val = d_open
                    tb_d_bit = 0
                else:
                    d_val = d_ext
                    tb_d_bit = 4
                D_curr[jm] = d_val

                jm_left = j - 1 - start_j_curr
                if 0 <= jm_left < band_width:
                    i_open = M_curr[jm_left] - gap_open - gap_extend
                    i_ext = I_curr[jm_left] - gap_extend
                else:
                    i_open = _INF
                    i_ext = _INF

                if i_open >= i_ext:
                    i_val = i_open
                    tb_i_bit = 0
                else:
                    i_val = i_ext
                    tb_i_bit = 8
                I_curr[jm] = i_val

                jm_top_left = j - 1 - start_j_prev
                m_prev_val = M_prev[jm_top_left] if 0 <= jm_top_left < band_width else _INF
                c2 = seq2[len2 - j] if reverse else seq2[j - 1]
                m_diag = (m_prev_val + int(matrix[c1, c2])) if m_prev_val > _INF // 2 else _INF

                best = m_diag
                tb_m_bits = 0
                if d_val > best:
                    best = d_val
                    tb_m_bits = 1
                if i_val > best:
                    best = i_val
                    tb_m_bits = 2

                if best < max_score - x_drop:
                    M_curr[jm] = _INF
                    tb[i, jm] = np.uint8(3 | tb_d_bit | tb_i_bit)
                else:
                    M_curr[jm] = best
                    tb[i, jm] = np.uint8(tb_m_bits | tb_d_bit | tb_i_bit)
                    if best > row_best_score:
                        row_best_score = best
                        row_best_j = j
                    if best > max_score:
                        max_score = best
                        max_i = i
                        max_j = j

        if row_best_score < max_score - x_drop:
            break

        best_diag = i - row_best_j
        if abs(best_diag - active_offset) > shift_tol:
            active_offset = best_diag

        M_prev, M_curr = M_curr, M_prev
        D_prev, D_curr = D_curr, D_prev

    if max_score <= 0:
        return 0, 0, 0, 0, 0, 0

    i, j = max_i, max_j
    matches = mismatches = gaps = state = 0

    while i > 0 or j > 0:
        start_j_curr = row_start_j[i]
        jm = j - start_j_curr
        if jm < 0 or jm >= band_width:
            break
        tb_val = tb[i, jm]

        if state == 0:
            tb_m = tb_val & 3
            if tb_m == 3:
                break
            elif tb_m == 0:
                c1 = seq1[len1 - i] if reverse else seq1[i - 1]
                c2 = seq2[len2 - j] if reverse else seq2[j - 1]
                if c1 == c2:
                    matches += 1
                else:
                    mismatches += 1
                i -= 1
                j -= 1
            else:
                state = int(tb_m)
        elif state == 1:
            tb_d = (tb_val >> 2) & 1
            gaps += 1
            i -= 1
            if tb_d == 0:
                state = 0
        elif state == 2:
            tb_i = (tb_val >> 3) & 1
            gaps += 1
            j -= 1
            if tb_i == 0:
                state = 0

    return int(max_score), matches, mismatches, gaps, max_i, max_j


@njit(cache=True, nogil=True)
def _wfa_align_pair(
    seq1: np.ndarray[tuple[int], np.dtype[np.uint8]],
    seq2: np.ndarray[tuple[int], np.dtype[np.uint8]],
    matrix: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    k: int,
    block_size: int,
    gap_open: int,
    gap_extend: int,
    x_drop: int,
    is_seeded: bool,
    offset: int,
    pinned: bool,
) -> tuple[int, int, int, int, int, int, int, int]:
    r"""Affine Wavefront Alignment (WFA2, Marco-Sola et al.) with greedy exact-match slide and adaptive pruning.

    When `pinned=True` (for inter-anchor gaps), aligns globally from `(0, 0)` to `(len1, len2)`.
    When `pinned=False`, trims negative-scoring overhangs via Kadane's scan along the optimal edit path,
    and falls back to `_block_gotoh_pair` if the edit penalty exceeds the wavefront budget `max_s`.
    """
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0 and len2 == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0
    if len1 == 0:
        if pinned:
            return -(gap_open + len2 * gap_extend), 0, 0, len2, 0, 0, 0, len2
        return 0, 0, 0, 0, 0, 0, 0, 0
    if len2 == 0:
        if pinned:
            return -(gap_open + len1 * gap_extend), 0, 0, len1, 0, len1, 0, 0
        return 0, 0, 0, 0, 0, 0, 0, 0

    # Fast path for identical-length sequences with 0 indels
    if len1 == len2:
        all_eq = True
        for idx in range(len1):
            if seq1[idx] != seq2[idx]:
                all_eq = False
                break
        if all_eq:
            sc = 0
            for idx in range(len1):
                sc += int(matrix[seq1[idx], seq2[idx]])
            return sc, len1, 0, 0, 0, len1, 0, len2

    # Wavefront unit penalties: mismatch = 2, gap_open = 3, gap_extend = 1
    x_wfa = 2
    o_wfa = 3
    e_wfa = 1
    oe_wfa = o_wfa + e_wfa

    max_s = min(max(192, (len1 + len2)), 768)
    if not pinned and abs(len1 - len2) > max_s // 2:
        return _block_gotoh_pair(seq1, seq2, matrix, k, block_size, gap_open, gap_extend, x_drop, is_seeded, offset)

    k_offset = len2 + 1
    n_diags = len1 + len2 + 3
    k_end = len1 - len2
    k_end_idx = k_end + k_offset

    M_wf = np.full((max_s + 1, n_diags), -2, dtype=np.int32)
    I_wf = np.full((max_s + 1, n_diags), -2, dtype=np.int32)
    D_wf = np.full((max_s + 1, n_diags), -2, dtype=np.int32)
    min_k_arr = np.zeros(max_s + 1, dtype=np.int32)
    max_k_arr = np.full(max_s + 1, -1, dtype=np.int32)

    # s = 0 initial wavefront at k = 0
    i0 = 0
    while i0 < len1 and i0 < len2 and seq1[i0] == seq2[i0]:
        i0 += 1
    M_wf[0, k_offset] = i0
    min_k_arr[0] = 0
    max_k_arr[0] = 0

    reached_s = -1
    if k_end == 0 and i0 == len1:
        reached_s = 0

    prune_lag = max(64, 2 * block_size + abs(len1 - len2))

    s = 1
    while reached_s < 0 and s <= max_s:
        lo_k = len1 + 1
        hi_k = -len2 - 1

        if s >= x_wfa and min_k_arr[s - x_wfa] <= max_k_arr[s - x_wfa]:
            if min_k_arr[s - x_wfa] < lo_k:
                lo_k = min_k_arr[s - x_wfa]
            if max_k_arr[s - x_wfa] > hi_k:
                hi_k = max_k_arr[s - x_wfa]
        if s >= oe_wfa and min_k_arr[s - oe_wfa] <= max_k_arr[s - oe_wfa]:
            if min_k_arr[s - oe_wfa] - 1 < lo_k:
                lo_k = min_k_arr[s - oe_wfa] - 1
            if max_k_arr[s - oe_wfa] + 1 > hi_k:
                hi_k = max_k_arr[s - oe_wfa] + 1
        if s >= e_wfa and min_k_arr[s - e_wfa] <= max_k_arr[s - e_wfa]:
            if min_k_arr[s - e_wfa] - 1 < lo_k:
                lo_k = min_k_arr[s - e_wfa] - 1
            if max_k_arr[s - e_wfa] + 1 > hi_k:
                hi_k = max_k_arr[s - e_wfa] + 1

        lo_k = max(-len2, lo_k)
        hi_k = min(len1, hi_k)
        if lo_k > hi_k:
            s += 1
            continue

        best_anti_diag = -1
        eff_lo_k = hi_k + 1
        eff_hi_k = lo_k - 1

        for diag_k in range(lo_k, hi_k + 1):
            k_idx = diag_k + k_offset

            # I_wf[s, k] from k + 1 (insertion in target: j increases by 1, i unchanged)
            ins_open = M_wf[s - oe_wfa, k_idx + 1] if s >= oe_wfa else -2
            ins_ext = I_wf[s - e_wfa, k_idx + 1] if s >= e_wfa else -2
            ins_val = ins_open if ins_open >= ins_ext else ins_ext
            if ins_val >= 0 and ins_val - diag_k <= len2:
                I_wf[s, k_idx] = ins_val
            else:
                ins_val = -2

            # D_wf[s, k] from k - 1 (deletion in target: i increases by 1)
            del_open = (M_wf[s - oe_wfa, k_idx - 1] + 1) if (s >= oe_wfa and M_wf[s - oe_wfa, k_idx - 1] >= 0) else -2
            del_ext = (D_wf[s - e_wfa, k_idx - 1] + 1) if (s >= e_wfa and D_wf[s - e_wfa, k_idx - 1] >= 0) else -2
            del_val = del_open if del_open >= del_ext else del_ext
            if del_val >= 0 and del_val <= len1:
                D_wf[s, k_idx] = del_val
            else:
                del_val = -2

            # M_wf[s, k] from mismatch at (s - x_wfa, k) or I_wf / D_wf
            sub_val = (M_wf[s - x_wfa, k_idx] + 1) if (s >= x_wfa and M_wf[s - x_wfa, k_idx] >= 0) else -2
            if sub_val > len1 or sub_val - diag_k > len2:
                sub_val = -2

            m_val = sub_val
            if ins_val > m_val:
                m_val = ins_val
            if del_val > m_val:
                m_val = del_val

            if m_val >= 0:
                cur_i = m_val
                cur_j = cur_i - diag_k
                if 0 <= cur_i <= len1 and 0 <= cur_j <= len2:
                    while cur_i < len1 and cur_j < len2 and seq1[cur_i] == seq2[cur_j]:
                        cur_i += 1
                        cur_j += 1
                    M_wf[s, k_idx] = cur_i
                    anti_d = cur_i + cur_j
                    if anti_d > best_anti_diag:
                        best_anti_diag = anti_d
                    if diag_k < eff_lo_k:
                        eff_lo_k = diag_k
                    if diag_k > eff_hi_k:
                        eff_hi_k = diag_k

        # Adaptive wavefront pruning at the boundaries
        if eff_lo_k <= eff_hi_k:
            while eff_lo_k < eff_hi_k:
                i_lo = M_wf[s, eff_lo_k + k_offset]
                if i_lo >= 0 and (2 * i_lo - eff_lo_k) >= best_anti_diag - prune_lag:
                    break
                eff_lo_k += 1
            while eff_hi_k > eff_lo_k:
                i_hi = M_wf[s, eff_hi_k + k_offset]
                if i_hi >= 0 and (2 * i_hi - eff_hi_k) >= best_anti_diag - prune_lag:
                    break
                eff_hi_k -= 1
            min_k_arr[s] = eff_lo_k
            max_k_arr[s] = eff_hi_k

        if M_wf[s, k_end_idx] == len1:
            reached_s = s
            break

        s += 1

    if reached_s < 0:
        # Exceeded wavefront budget; fallback to adaptive block Gotoh
        if pinned:
            sc_b, m_b, mm_b, g_b, qs_b, qe_b, ts_b, te_b = _block_gotoh_pair(
                seq1, seq2, matrix, k, block_size, gap_open, gap_extend, 0, False, 0
            )
            rem_q = (len1 - qe_b) + qs_b
            rem_t = (len2 - te_b) + ts_b
            if rem_q > 0 or rem_t > 0:
                g_b += rem_q + rem_t
                sc_b -= gap_open + (rem_q + rem_t) * gap_extend
            return sc_b, m_b, mm_b, g_b, 0, len1, 0, len2
        return _block_gotoh_pair(seq1, seq2, matrix, k, block_size, gap_open, gap_extend, x_drop, is_seeded, offset)

    # Traceback through wavefronts from (reached_s, k_end) to (0, 0)
    max_ops = len1 + len2 + 1
    op_type = np.empty(max_ops, dtype=np.uint8)  # 0=match, 1=mismatch, 2=del(dq=1,dt=0), 3=ins(dq=0,dt=1)
    op_score = np.empty(max_ops, dtype=np.int32)
    n_ops = 0

    cur_s = reached_s
    cur_k = k_end
    state = 0  # 0=M, 1=D, 2=I

    while cur_s >= 0:
        k_idx = cur_k + k_offset
        if state == 0:
            cur_i = int(M_wf[cur_s, k_idx])
            cur_j = cur_i - cur_k

            ins_val = int(I_wf[cur_s, k_idx])
            del_val = int(D_wf[cur_s, k_idx])
            sub_val = (
                (int(M_wf[cur_s - x_wfa, k_idx]) + 1) if (cur_s >= x_wfa and M_wf[cur_s - x_wfa, k_idx] >= 0) else -2
            )
            if sub_val > len1 or sub_val - cur_k > len2:
                sub_val = -2

            i_pre = 0 if (cur_s == 0 and cur_k == 0) else sub_val
            next_state = 0
            if ins_val > i_pre:
                i_pre = ins_val
                next_state = 2
            if del_val > i_pre:
                i_pre = del_val
                next_state = 1

            slide_len = cur_i - i_pre
            for r in range(slide_len):
                ii = cur_i - 1 - r
                jj = cur_j - 1 - r
                op_type[n_ops] = 0 if seq1[ii] == seq2[jj] else 1
                op_score[n_ops] = int(matrix[seq1[ii], seq2[jj]])
                n_ops += 1

            if cur_s == 0 and cur_k == 0:
                break

            if next_state == 2:
                state = 2
            elif next_state == 1:
                state = 1
            else:
                ii = i_pre - 1
                jj = ii - cur_k
                op_type[n_ops] = 0 if seq1[ii] == seq2[jj] else 1
                op_score[n_ops] = int(matrix[seq1[ii], seq2[jj]])
                n_ops += 1
                cur_s -= x_wfa
        elif state == 1:
            del_cur = int(D_wf[cur_s, k_idx])
            del_ext = (
                (int(D_wf[cur_s - e_wfa, k_idx - 1]) + 1)
                if (cur_s >= e_wfa and D_wf[cur_s - e_wfa, k_idx - 1] >= 0)
                else -2
            )
            op_type[n_ops] = 2
            if del_ext == del_cur:
                op_score[n_ops] = -gap_extend
                n_ops += 1
                cur_s -= e_wfa
                cur_k -= 1
            else:
                op_score[n_ops] = -(gap_open + gap_extend)
                n_ops += 1
                cur_s -= oe_wfa
                cur_k -= 1
                state = 0
        elif state == 2:
            ins_cur = int(I_wf[cur_s, k_idx])
            ins_ext = int(I_wf[cur_s - e_wfa, k_idx + 1]) if cur_s >= e_wfa else -2
            op_type[n_ops] = 3
            if ins_ext == ins_cur:
                op_score[n_ops] = -gap_extend
                n_ops += 1
                cur_s -= e_wfa
                cur_k += 1
            else:
                op_score[n_ops] = -(gap_open + gap_extend)
                n_ops += 1
                cur_s -= oe_wfa
                cur_k += 1
                state = 0

    if pinned:
        tot_sc = 0
        matches = 0
        mismatches = 0
        gaps = 0
        for idx in range(n_ops):
            tot_sc += int(op_score[idx])
            t_op = op_type[idx]
            if t_op == 0:
                matches += 1
            elif t_op == 1:
                mismatches += 1
            else:
                gaps += 1
        return tot_sc, matches, mismatches, gaps, 0, len1, 0, len2

    # Unpinned mode: Kadane's scan along forward edit operations (op indices n_ops - 1 down to 0)
    best_sc = 0
    best_start_op = n_ops
    best_end_op = n_ops
    cur_sc = 0
    cur_start_op = n_ops - 1

    for rev_idx in range(n_ops - 1, -1, -1):
        val = int(op_score[rev_idx])
        if cur_sc + val <= 0:
            cur_sc = 0
            cur_start_op = rev_idx - 1
        else:
            cur_sc += val
            if cur_sc > best_sc:
                best_sc = cur_sc
                best_start_op = cur_start_op
                best_end_op = rev_idx

    if best_sc <= 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    q_pos = 0
    t_pos = 0
    q_start = 0
    t_start = 0
    q_end = 0
    t_end = 0
    matches = 0
    mismatches = 0
    gaps = 0

    for rev_idx in range(n_ops - 1, -1, -1):
        if rev_idx == best_start_op:
            q_start = q_pos
            t_start = t_pos
        t_op = op_type[rev_idx]
        in_seg = best_end_op <= rev_idx <= best_start_op
        if t_op == 0:
            q_pos += 1
            t_pos += 1
            if in_seg:
                matches += 1
        elif t_op == 1:
            q_pos += 1
            t_pos += 1
            if in_seg:
                mismatches += 1
        elif t_op == 2:
            q_pos += 1
            if in_seg:
                gaps += 1
        else:
            t_pos += 1
            if in_seg:
                gaps += 1
        if rev_idx == best_end_op:
            q_end = q_pos
            t_end = t_pos

    return best_sc, matches, mismatches, gaps, q_start, q_end, t_start, t_end


@njit(parallel=True, cache=True, nogil=True)
def _batched_banded_gotoh(
    q_data: np.ndarray[tuple[int], np.dtype[np.uint8]],
    q_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_data: np.ndarray[tuple[int], np.dtype[np.uint8]],
    t_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    matrix: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    k: int,
    gap_open: int,
    gap_extend: int,
    is_seeded: bool,
    diagonal_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_scores: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_matches: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_mismatches: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_gaps: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_q_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_q_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_t_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_t_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> None:
    r"""Numba parallel kernel for batched static-band Smith-Waterman-Gotoh pairwise alignment."""
    for idx in prange(len(q_offsets)):  # type: ignore[not-iterable]
        seq1 = q_data[q_offsets[idx] : q_offsets[idx] + q_lengths[idx]]
        seq2 = t_data[t_offsets[idx] : t_offsets[idx] + t_lengths[idx]]
        sc, m, mm, g, qs, qe, ts, te = _banded_gotoh_pair(
            seq1,
            seq2,
            matrix,
            k,
            gap_open,
            gap_extend,
            is_seeded,
            int(diagonal_offsets[idx]),
        )
        out_scores[idx] = sc
        out_matches[idx] = m
        out_mismatches[idx] = mm
        out_gaps[idx] = g
        out_q_starts[idx] = qs
        out_q_ends[idx] = qe
        out_t_starts[idx] = ts
        out_t_ends[idx] = te


@njit(parallel=True, cache=True, nogil=True)
def _batched_block_gotoh(
    q_data: np.ndarray[tuple[int], np.dtype[np.uint8]],
    q_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_data: np.ndarray[tuple[int], np.dtype[np.uint8]],
    t_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    matrix: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    k: int,
    block_size: int,
    gap_open: int,
    gap_extend: int,
    x_drop: int,
    is_seeded: bool,
    diagonal_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_scores: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_matches: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_mismatches: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_gaps: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_q_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_q_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_t_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_t_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> None:
    r"""Numba parallel kernel for batched adaptive block-shifting Smith-Waterman-Gotoh pairwise alignment."""
    for idx in prange(len(q_offsets)):  # type: ignore[not-iterable]
        seq1 = q_data[q_offsets[idx] : q_offsets[idx] + q_lengths[idx]]
        seq2 = t_data[t_offsets[idx] : t_offsets[idx] + t_lengths[idx]]
        sc, m, mm, g, qs, qe, ts, te = _block_gotoh_pair(
            seq1,
            seq2,
            matrix,
            k,
            block_size,
            gap_open,
            gap_extend,
            x_drop,
            is_seeded,
            int(diagonal_offsets[idx]),
        )
        out_scores[idx] = sc
        out_matches[idx] = m
        out_mismatches[idx] = mm
        out_gaps[idx] = g
        out_q_starts[idx] = qs
        out_q_ends[idx] = qe
        out_t_starts[idx] = ts
        out_t_ends[idx] = te


@njit(parallel=True, cache=True, nogil=True)
def _batched_wfa(
    q_data: np.ndarray[tuple[int], np.dtype[np.uint8]],
    q_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_data: np.ndarray[tuple[int], np.dtype[np.uint8]],
    t_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    matrix: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    k: int,
    block_size: int,
    gap_open: int,
    gap_extend: int,
    x_drop: int,
    is_seeded: bool,
    diagonal_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_scores: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_matches: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_mismatches: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_gaps: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_q_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_q_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_t_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_t_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> None:
    r"""Numba parallel kernel for batched Affine Wavefront Alignment (WFA2)."""
    for idx in prange(len(q_offsets)):  # type: ignore[not-iterable]
        seq1 = q_data[q_offsets[idx] : q_offsets[idx] + q_lengths[idx]]
        seq2 = t_data[t_offsets[idx] : t_offsets[idx] + t_lengths[idx]]
        sc, m, mm, g, qs, qe, ts, te = _wfa_align_pair(
            seq1,
            seq2,
            matrix,
            k,
            block_size,
            gap_open,
            gap_extend,
            x_drop,
            is_seeded,
            int(diagonal_offsets[idx]),
            False,
        )
        out_scores[idx] = sc
        out_matches[idx] = m
        out_mismatches[idx] = mm
        out_gaps[idx] = g
        out_q_starts[idx] = qs
        out_q_ends[idx] = qe
        out_t_starts[idx] = ts
        out_t_ends[idx] = te


@njit(parallel=True, cache=True, nogil=True)
def _align_chains_kernel(
    q_data: np.ndarray[tuple[int], np.dtype[np.uint8]],
    q_seq_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_seq_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_data: np.ndarray[tuple[int], np.dtype[np.uint8]],
    t_seq_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_seq_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    chain_q_indices: np.ndarray[tuple[int], np.dtype[np.uint32]],
    chain_t_indices: np.ndarray[tuple[int], np.dtype[np.uint32]],
    chain_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    chain_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    anchor_q_pos: np.ndarray[tuple[int], np.dtype[np.int32]],
    anchor_t_pos: np.ndarray[tuple[int], np.dtype[np.int32]],
    anchor_spans: np.ndarray[tuple[int], np.dtype[np.uint16]],
    matrix: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    block_size: int,
    gap_open: int,
    gap_extend: int,
    x_drop: int,
    out_scores: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_matches: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_mismatches: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_gaps: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_q_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_q_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_t_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_t_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> None:
    r"""Piecewise anchor-guided alignment using WFA2 for inter-anchor gaps and block X-drop at chain ends."""
    n_chains = len(chain_offsets)
    for c_idx in prange(n_chains):  # type: ignore[not-iterable]
        q_id = int(chain_q_indices[c_idx])
        t_id = int(chain_t_indices[c_idx])
        c_start = int(chain_offsets[c_idx])
        c_len = int(chain_lengths[c_idx])

        seq1 = q_data[q_seq_offsets[q_id] : q_seq_offsets[q_id] + q_seq_lengths[q_id]]
        seq2 = t_data[t_seq_offsets[t_id] : t_seq_offsets[t_id] + t_seq_lengths[t_id]]
        len1, len2 = len(seq1), len(seq2)

        if c_len == 0 or len1 == 0 or len2 == 0:
            out_scores[c_idx] = 0
            out_matches[c_idx] = 0
            out_mismatches[c_idx] = 0
            out_gaps[c_idx] = 0
            out_q_starts[c_idx] = 0
            out_q_ends[c_idx] = 0
            out_t_starts[c_idx] = 0
            out_t_ends[c_idx] = 0
            continue

        q0 = min(max(0, int(anchor_q_pos[c_start])), len1)
        t0 = min(max(0, int(anchor_t_pos[c_start])), len2)

        # 1. Left extension before first anchor (q0, t0)
        if q0 > 0 and t0 > 0:
            l_sc, l_m, l_mm, l_g, ext_q_l, ext_t_l = _block_xdrop_extend(
                seq1[:q0], seq2[:t0], matrix, block_size, gap_open, gap_extend, x_drop, True
            )
            q_start = q0 - ext_q_l
            t_start = t0 - ext_t_l
        else:
            l_sc = l_m = l_mm = l_g = 0
            q_start = q0
            t_start = t0

        total_score = l_sc
        total_matches = l_m
        total_mismatches = l_mm
        total_gaps = l_g

        cur_q = q0
        cur_t = t0

        # 2. Traverse anchors and align inter-anchor gaps with pinned WFA2
        for a_idx in range(c_start, c_start + c_len):
            a_q = min(max(0, int(anchor_q_pos[a_idx])), len1)
            a_t = min(max(0, int(anchor_t_pos[a_idx])), len2)
            a_span = int(anchor_spans[a_idx])
            if a_q + a_span > len1:
                a_span = max(0, len1 - a_q)
            if a_t + a_span > len2:
                a_span = max(0, len2 - a_t)

            if a_q > cur_q or a_t > cur_t:
                if a_q > cur_q and a_t > cur_t:
                    g_sc, g_m, g_mm, g_g, _, _, _, _ = _wfa_align_pair(
                        seq1[cur_q:a_q],
                        seq2[cur_t:a_t],
                        matrix,
                        block_size,
                        block_size,
                        gap_open,
                        gap_extend,
                        x_drop,
                        False,
                        0,
                        True,
                    )
                    total_score += g_sc
                    total_matches += g_m
                    total_mismatches += g_mm
                    total_gaps += g_g
                elif a_q > cur_q:
                    gap_len = a_q - cur_q
                    total_gaps += gap_len
                    total_score -= gap_open + gap_len * gap_extend
                else:
                    gap_len = a_t - cur_t
                    total_gaps += gap_len
                    total_score -= gap_open + gap_len * gap_extend
                cur_q = a_q
                cur_t = a_t

            end_q = a_q + a_span
            end_t = a_t + a_span
            if end_q > cur_q and end_t > cur_t:
                if (cur_q - a_q) == (cur_t - a_t):
                    overlap = max(0, cur_q - a_q)
                    for r in range(overlap, a_span):
                        c1 = seq1[a_q + r]
                        c2 = seq2[a_t + r]
                        total_score += int(matrix[c1, c2])
                        if c1 == c2:
                            total_matches += 1
                        else:
                            total_mismatches += 1
                    cur_q = end_q
                    cur_t = end_t
                else:
                    g_sc, g_m, g_mm, g_g, _, _, _, _ = _wfa_align_pair(
                        seq1[cur_q:end_q],
                        seq2[cur_t:end_t],
                        matrix,
                        block_size,
                        block_size,
                        gap_open,
                        gap_extend,
                        x_drop,
                        False,
                        0,
                        True,
                    )
                    total_score += g_sc
                    total_matches += g_m
                    total_mismatches += g_mm
                    total_gaps += g_g
                    cur_q = end_q
                    cur_t = end_t

        # 3. Right extension after last anchor (cur_q, cur_t)
        if cur_q < len1 and cur_t < len2:
            r_sc, r_m, r_mm, r_g, ext_q_r, ext_t_r = _block_xdrop_extend(
                seq1[cur_q:], seq2[cur_t:], matrix, block_size, gap_open, gap_extend, x_drop, False
            )
            total_score += r_sc
            total_matches += r_m
            total_mismatches += r_mm
            total_gaps += r_g
            q_end = cur_q + ext_q_r
            t_end = cur_t + ext_t_r
        else:
            q_end = cur_q
            t_end = cur_t

        out_scores[c_idx] = max(0, total_score)
        out_matches[c_idx] = total_matches
        out_mismatches[c_idx] = total_mismatches
        out_gaps[c_idx] = total_gaps
        out_q_starts[c_idx] = q_start
        out_q_ends[c_idx] = q_end
        out_t_starts[c_idx] = t_start
        out_t_ends[c_idx] = t_end
