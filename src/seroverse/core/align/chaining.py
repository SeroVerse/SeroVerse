r"""Collinear anchor chaining (`minimap2`-style concave-gap DP) and ragged SoA containers.

This module provides an Object-Oriented and Data-Oriented Structure-of-Arrays (SoA)
collinear anchor chaining system inspired by `minimap2` (Li, 2018):
- [`Chain`][seroverse.core.align.chaining.Chain]: Immutable scalar view of a single collinear chain.
- [`Chains`][seroverse.core.align.chaining.Chains]: Ragged SoA batch container (`RaggedArrayContainer`)
  storing variable-length anchor chains across sequence pairs.
- [`Chainer`][seroverse.core.align.chaining.Chainer]: Parallel Numba JIT collinear chaining and
  piecewise WFA2 + adaptive block X-drop base-alignment engine.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numba import njit, prange

from seroverse.core.align.pairwise import PairwiseAligner, PairwiseAlignments
from seroverse.core.align.seeds import Seeds
from seroverse.core.collections import BoolArray, Int32Array, RaggedArrayContainer, ScalarView, UInt16Array, UInt32Array
from seroverse.core.seq.kmers import BaseKmerIndex
from seroverse.core.seq.seq import Sequences


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Chain(ScalarView):
    r"""Immutable scalar view of a single collinear anchor chain between a query and target sequence.

    Attributes:
        query_index (int): Index of the query sequence.
        target_index (int): Index of the target sequence.
        score (int): Collinear chaining dynamic programming score.
        q_start (int): 0-based start coordinate of the first anchor on the query sequence.
        q_end (int): 0-based exclusive end coordinate of the last anchor on the query sequence.
        t_start (int): 0-based start coordinate of the first anchor on the target sequence.
        t_end (int): 0-based exclusive end coordinate of the last anchor on the target sequence.
        is_primary (bool): True if classified as a primary chain on the query sequence.
        q_pos (Int32Array): 1D array of anchor query start coordinates.
        t_pos (Int32Array): 1D array of anchor target start coordinates.
        spans (UInt16Array): 1D array of anchor match spans (`k`).
    """

    query_index: int
    target_index: int
    score: int
    q_start: int
    q_end: int
    t_start: int
    t_end: int
    is_primary: bool
    q_pos: Int32Array
    t_pos: Int32Array
    spans: UInt16Array

    def __len__(self) -> int:
        r"""Return the number of anchors in this chain."""
        return len(self.q_pos)

    @property
    def offset(self) -> int:
        r"""Return the initial diagonal offset `q_start - t_start` of this chain."""
        return int(self.q_start - self.t_start)

    @property
    def seeds(self) -> Seeds:
        r"""Return the anchors belonging to this chain as a [`Seeds`][seroverse.core.align.seeds.Seeds] batch."""
        n = len(self.q_pos)
        return Seeds(
            query_indices=np.full(n, self.query_index, dtype=np.uint32),
            target_indices=np.full(n, self.target_index, dtype=np.uint32),
            scores=self.spans.astype(np.uint32),
            offsets=(self.q_pos - self.t_pos).astype(np.int32),
            q_pos=self.q_pos,
            t_pos=self.t_pos,
            spans=self.spans,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Chains(RaggedArrayContainer[Chain]):
    r"""Ragged Structure-of-Arrays (SoA) batch container for collinear anchor chains.

    Stores per-chain summary arrays (`query_indices`, `target_indices`, `scores`, `q_starts`,
    `q_ends`, `t_starts`, `t_ends`, `is_primary`) alongside CSR-style `offsets` and `lengths`
    indexing into flat concatenated anchor arrays (`anchor_q_pos`, `anchor_t_pos`, `anchor_spans`).

    Attributes:
        query_indices (UInt32Array): 1D array of query sequence indices.
        target_indices (UInt32Array): 1D array of target sequence indices.
        scores (Int32Array): 1D array of chaining DP scores.
        q_starts (Int32Array): 1D array of chain query start coordinates.
        q_ends (Int32Array): 1D array of chain query end coordinates.
        t_starts (Int32Array): 1D array of chain target start coordinates.
        t_ends (Int32Array): 1D array of chain target end coordinates.
        is_primary (BoolArray): 1D boolean array of primary chain flags.
        offsets (Int32Array): 1D array of start indices into anchor arrays.
        lengths (Int32Array): 1D array of anchor counts per chain.
        anchor_q_pos (Int32Array): Concatenated 1D array of anchor query positions.
        anchor_t_pos (Int32Array): Concatenated 1D array of anchor target positions.
        anchor_spans (UInt16Array): Concatenated 1D array of anchor spans.
    """

    _ragged_fields: ClassVar[tuple[str, ...]] = ("anchor_q_pos", "anchor_t_pos", "anchor_spans")

    query_indices: UInt32Array
    target_indices: UInt32Array
    scores: Int32Array
    q_starts: Int32Array
    q_ends: Int32Array
    t_starts: Int32Array
    t_ends: Int32Array
    is_primary: BoolArray
    offsets: Int32Array
    lengths: Int32Array
    anchor_q_pos: Int32Array
    anchor_t_pos: Int32Array
    anchor_spans: UInt16Array

    @property
    def total_anchors(self) -> int:
        r"""Return the total number of anchors across all chains."""
        return len(self.anchor_q_pos)

    @property
    def diagonal_offsets(self) -> Int32Array:
        r"""Return the 1D array of initial diagonal offsets `q_starts - t_starts` for each chain."""
        return (self.q_starts - self.t_starts).astype(np.int32)

    def primary(self) -> Chains:
        r"""Return only primary chains (`self[self.is_primary]`)."""
        return self[self.is_primary]

    def top_chains(self) -> Chains:
        r"""Select the single highest-scoring chain for each unique query sequence."""
        if len(self) == 0:
            return self
        sort_order = np.lexsort((-self.scores.astype(np.int64), self.query_indices))
        sorted_queries = self.query_indices[sort_order]
        unique_mask = np.empty(len(sorted_queries), dtype=np.bool_)
        unique_mask[0] = True
        unique_mask[1:] = sorted_queries[1:] != sorted_queries[:-1]
        best_indices = np.sort(sort_order[unique_mask])
        return self[best_indices]

    def to_seeds(self) -> Seeds:
        r"""Convert chains into pair-level consensus [`Seeds`][seroverse.core.align.seeds.Seeds]."""
        if len(self) == 0:
            return Seeds.empty()
        spans = np.minimum(65535, np.maximum(1, self.q_ends - self.q_starts)).astype(np.uint16)
        return Seeds(
            query_indices=self.query_indices,
            target_indices=self.target_indices,
            scores=np.maximum(0, self.scores).astype(np.uint32),
            offsets=(self.q_starts - self.t_starts).astype(np.int32),
            q_pos=self.q_starts,
            t_pos=self.t_starts,
            spans=spans,
        )

    def extract_sequences(self, queries: Sequences, targets: Sequences) -> tuple[Sequences, Sequences]:
        r"""Extract parallel batches of query and target sequences mapped by this `Chains` batch."""
        return queries[self.query_indices], targets[self.target_indices]


@dataclass(frozen=True, slots=True)
class Chainer:
    r"""Collinear anchor chaining and piecewise alignment engine (`minimap2`-style concave-gap DP).

    Chains unaggregated [`Seeds`][seroverse.core.align.seeds.Seeds] (produced by
    `index.hits(queries, aggregate=False)`) into collinear [`Chains`][seroverse.core.align.chaining.Chains]
    using a concave gap penalty function and `max_skip` predecessor pruning, classifies primary vs
    secondary chains, and performs piecewise base-level alignment via WFA2 + adaptive block X-drop.

    Attributes:
        max_dist_x (int): Maximum target distance between consecutive anchors in a chain. Defaults to 5000.
        max_dist_y (int): Maximum query distance between consecutive anchors in a chain. Defaults to 5000.
        max_band (int): Maximum diagonal shift $|(q_i - q_j) - (t_i - t_j)|$ between chained anchors. Defaults to 500.
        min_cnt (int): Minimum number of anchors required to form a valid chain. Defaults to 2.
        min_score (int): Minimum chaining DP score required to emit a chain. Defaults to 20.
        gap_scale (float): Concave gap penalty scaling multiplier. Defaults to 1.0.
        max_skip (int): Heuristic predecessor pruning threshold. Defaults to 25.
        mask_level (float): Minimum query overlap ratio to mask a lower-scoring chain as secondary. Defaults to 0.5.
    """

    max_dist_x: int = 5000
    max_dist_y: int = 5000
    max_band: int = 500
    min_cnt: int = 2
    min_score: int = 20
    gap_scale: float = 1.0
    max_skip: int = 25
    mask_level: float = 0.5

    def __call__(self, seeds: Seeds) -> Chains:
        r"""Chain unaggregated anchor [`Seeds`][seroverse.core.align.seeds.Seeds] into collinear [`Chains`][seroverse.core.align.chaining.Chains].

        Args:
            seeds (Seeds): Unaggregated anchor hits from `index.hits(queries, aggregate=False)` or custom anchors.

        Returns:
            Chains: Ragged SoA container of collinear chains sorted by `(query_index, -score)`.
        """
        n_anchors = len(seeds)
        if n_anchors == 0:
            return Chains.empty()

        if not _is_sorted_anchors(seeds.query_indices, seeds.target_indices, seeds.t_pos, seeds.q_pos):
            order = np.lexsort((seeds.q_pos, seeds.t_pos, seeds.target_indices, seeds.query_indices))
            seeds = seeds[order]

        pair_starts, pair_ends = _find_pair_runs(seeds.query_indices, seeds.target_indices)
        n_pairs = len(pair_starts)
        if n_pairs == 0:
            return Chains.empty()

        tmp_c_score = np.empty(n_anchors, dtype=np.int32)
        tmp_c_len = np.empty(n_anchors, dtype=np.int32)
        tmp_c_a_start = np.empty(n_anchors, dtype=np.int32)
        tmp_anchor_idx = np.empty(n_anchors, dtype=np.int32)
        pair_chain_counts = np.zeros(n_pairs, dtype=np.int32)
        pair_anchor_counts = np.zeros(n_pairs, dtype=np.int32)

        _chain_pairs_kernel(
            seeds.q_pos,
            seeds.t_pos,
            seeds.spans,
            pair_starts,
            pair_ends,
            int(self.max_dist_x),
            int(self.max_dist_y),
            int(self.max_band),
            int(self.min_cnt),
            int(self.min_score),
            float(self.gap_scale),
            int(self.max_skip),
            tmp_c_score,
            tmp_c_len,
            tmp_c_a_start,
            tmp_anchor_idx,
            pair_chain_counts,
            pair_anchor_counts,
        )

        total_chains = int(np.sum(pair_chain_counts))
        total_chained_anchors = int(np.sum(pair_anchor_counts))
        if total_chains == 0:
            return Chains.empty()

        chain_indptr = Chains.lengths_to_offsets(pair_chain_counts)
        anchor_indptr = Chains.lengths_to_offsets(pair_anchor_counts)

        out_q = np.empty(total_chains, dtype=np.uint32)
        out_t = np.empty(total_chains, dtype=np.uint32)
        out_scores = np.empty(total_chains, dtype=np.int32)
        out_q_starts = np.empty(total_chains, dtype=np.int32)
        out_q_ends = np.empty(total_chains, dtype=np.int32)
        out_t_starts = np.empty(total_chains, dtype=np.int32)
        out_t_ends = np.empty(total_chains, dtype=np.int32)
        out_offsets = np.empty(total_chains, dtype=np.int32)
        out_lengths = np.empty(total_chains, dtype=np.int32)
        out_aq = np.empty(total_chained_anchors, dtype=np.int32)
        out_at = np.empty(total_chained_anchors, dtype=np.int32)
        out_asp = np.empty(total_chained_anchors, dtype=np.uint16)

        _compact_chains_kernel(
            seeds.query_indices,
            seeds.target_indices,
            seeds.q_pos,
            seeds.t_pos,
            seeds.spans,
            pair_starts,
            pair_chain_counts,
            chain_indptr,
            anchor_indptr,
            tmp_c_score,
            tmp_c_len,
            tmp_c_a_start,
            tmp_anchor_idx,
            out_q,
            out_t,
            out_scores,
            out_q_starts,
            out_q_ends,
            out_t_starts,
            out_t_ends,
            out_offsets,
            out_lengths,
            out_aq,
            out_at,
            out_asp,
        )

        out_primary = _classify_primary_chains_kernel(
            out_q,
            out_scores,
            out_q_starts,
            out_q_ends,
            float(self.mask_level),
        )

        return Chains(
            query_indices=out_q,
            target_indices=out_t,
            scores=out_scores,
            q_starts=out_q_starts,
            q_ends=out_q_ends,
            t_starts=out_t_starts,
            t_ends=out_t_ends,
            is_primary=out_primary,
            offsets=out_offsets,
            lengths=out_lengths,
            anchor_q_pos=out_aq,
            anchor_t_pos=out_at,
            anchor_spans=out_asp,
        )

    def align(
        self,
        queries: Sequences,
        targets: Sequences,
        chains: Chains,
        aligner: PairwiseAligner | None = None,
    ) -> PairwiseAlignments:
        r"""Perform piecewise anchor-guided base alignment for each chain in `chains`.

        Uses WFA2 for inter-anchor gaps and adaptive block X-drop extension at chain ends.

        Args:
            queries (Sequences): Query sequences collection.
            targets (Sequences): Target sequences collection.
            chains (Chains): Collinear chains to align at the base/residue level.
            aligner (PairwiseAligner | None): Optional `PairwiseAligner` providing scoring matrix and gap penalties.

        Returns:
            PairwiseAlignments: Base-level alignment statistics parallel to `chains`.
        """
        eff_aligner = aligner if aligner is not None else PairwiseAligner()
        return eff_aligner.align_chains(queries, targets, chains)

    def search(
        self,
        queries: Sequences,
        index: BaseKmerIndex,
        targets: Sequences,
        aligner: PairwiseAligner | None = None,
        *,
        min_seed_score: int = 1,
        primary_only: bool = True,
    ) -> tuple[Chains, PairwiseAlignments]:
        r"""Execute a full `minimap2`-style seed-chain-align search pipeline.

        Extracts unaggregated anchors via `index.hits(queries, min_score=min_seed_score, aggregate=False)`,
        chains them with concave-gap DP, optionally filters to primary chains, and performs piecewise
        WFA2 + adaptive block X-drop base alignment.

        Args:
            queries (Sequences): Query sequences.
            index (BaseKmerIndex): Sorted k-mer / syncmer / randstrobe index built on `targets`.
            targets (Sequences): Target sequences.
            aligner (PairwiseAligner | None): Optional `PairwiseAligner` for scoring parameters.
                Defaults to `PairwiseAligner(matrix=index._default_matrix())`.
            min_seed_score (int): Minimum anchor hits per `(query, target)` pair before chaining. Defaults to 1.
            primary_only (bool): If True (default), restricts output to primary chains (`chains.primary()`).

        Returns:
            tuple[Chains, PairwiseAlignments]: Parallel `(chains, alignments)` for all discovered homologous chains.
        """
        eff_aligner = aligner if aligner is not None else PairwiseAligner(matrix=index._default_matrix())
        raw_seeds = index.hits(queries, min_score=min_seed_score, aggregate=False)
        if len(raw_seeds) == 0:
            return Chains.empty(), PairwiseAlignments.empty()

        chains = self(raw_seeds)
        if primary_only:
            chains = chains.primary()
        if len(chains) == 0:
            return chains, PairwiseAlignments.empty()

        alignments = self.align(queries, targets, chains, aligner=eff_aligner)
        return chains, alignments


# Kernels --------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _is_sorted_anchors(
    q_indices: np.ndarray[tuple[int], np.dtype[np.uint32]],
    t_indices: np.ndarray[tuple[int], np.dtype[np.uint32]],
    t_pos: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_pos: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> bool:
    r"""Check whether anchors are sorted by `(query_index, target_index, t_pos, q_pos)`."""
    n = len(q_indices)
    for i in range(1, n):
        q_prev, q_curr = q_indices[i - 1], q_indices[i]
        if q_curr < q_prev:
            return False
        if q_curr == q_prev:
            t_prev, t_curr = t_indices[i - 1], t_indices[i]
            if t_curr < t_prev:
                return False
            if t_curr == t_prev:
                tp_prev, tp_curr = t_pos[i - 1], t_pos[i]
                if tp_curr < tp_prev:
                    return False
                if tp_curr == tp_prev and q_pos[i] < q_pos[i - 1]:
                    return False
    return True


@njit(cache=True, nogil=True)
def _find_pair_runs(
    q_indices: np.ndarray[tuple[int], np.dtype[np.uint32]],
    t_indices: np.ndarray[tuple[int], np.dtype[np.uint32]],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
]:
    r"""Find contiguous `[start, end)` run boundaries for each `(query_index, target_index)` pair."""
    n = len(q_indices)
    if n == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    n_runs = 1
    for i in range(1, n):
        if q_indices[i] != q_indices[i - 1] or t_indices[i] != t_indices[i - 1]:
            n_runs += 1

    starts = np.empty(n_runs, dtype=np.int32)
    ends = np.empty(n_runs, dtype=np.int32)
    starts[0] = 0
    r = 0
    for i in range(1, n):
        if q_indices[i] != q_indices[i - 1] or t_indices[i] != t_indices[i - 1]:
            ends[r] = i
            r += 1
            starts[r] = i
    ends[r] = n
    return starts, ends


@njit(parallel=True, cache=True, nogil=True)
def _chain_pairs_kernel(
    q_pos: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_pos: np.ndarray[tuple[int], np.dtype[np.int32]],
    spans: np.ndarray[tuple[int], np.dtype[np.uint16]],
    pair_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    pair_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
    max_dist_x: int,
    max_dist_y: int,
    max_band: int,
    min_cnt: int,
    min_score: int,
    gap_scale: float,
    max_skip: int,
    tmp_c_score: np.ndarray[tuple[int], np.dtype[np.int32]],
    tmp_c_len: np.ndarray[tuple[int], np.dtype[np.int32]],
    tmp_c_a_start: np.ndarray[tuple[int], np.dtype[np.int32]],
    tmp_anchor_idx: np.ndarray[tuple[int], np.dtype[np.int32]],
    pair_chain_counts: np.ndarray[tuple[int], np.dtype[np.int32]],
    pair_anchor_counts: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> None:
    r"""Parallel `minimap2`-style concave-gap collinear chaining DP and non-overlapping chain backtracking."""
    n_anchors = len(q_pos)
    n_pairs = len(pair_starts)

    dp_f = np.empty(n_anchors, dtype=np.int32)
    dp_p = np.full(n_anchors, -1, dtype=np.int32)
    dp_v = np.zeros(n_anchors, dtype=np.int32)
    used = np.zeros(n_anchors, dtype=np.uint8)

    for p_idx in prange(n_pairs):  # type: ignore[not-iterable]
        start = int(pair_starts[p_idx])
        end = int(pair_ends[p_idx])
        m = end - start
        if m < min_cnt:
            continue

        # 1. Forward concave-gap chaining DP
        for i in range(start, end):
            qi = int(q_pos[i])
            ti = int(t_pos[i])
            wi = max(1, int(spans[i]))
            max_f = wi
            max_j = -1
            n_skip = 0

            for j in range(i - 1, start - 1, -1):
                dx = ti - int(t_pos[j])
                if dx > max_dist_x:
                    break
                if dx <= 0:
                    continue
                dy = qi - int(q_pos[j])
                if dy <= 0 or dy > max_dist_y:
                    continue

                l_gap = abs(dy - dx)
                if l_gap > max_band:
                    continue

                min_d = dx if dx < dy else dy
                alpha = min_d if min_d < wi else wi

                if l_gap == 0:
                    beta = 0
                else:
                    log_l = math.log2(float(l_gap)) if l_gap >= 2 else 1.0
                    beta = int(gap_scale * (0.01 * float(wi) * float(l_gap) + 0.5 * log_l))

                sc = int(dp_f[j]) + alpha - beta
                if sc > max_f:
                    max_f = sc
                    max_j = j
                    if n_skip > 0:
                        n_skip -= 1
                elif dp_v[j] == dp_f[j]:
                    n_skip += 1
                    if n_skip > max_skip:
                        break

            dp_f[i] = max_f
            dp_p[i] = max_j
            dp_v[i] = max_f
            if max_j >= 0 and dp_v[max_j] < max_f:
                dp_v[max_j] = max_f

        # 2. Sort anchor indices in [start, end) by descending dp_f score for backtracking
        order = np.argsort(-dp_f[start:end])

        c_write = start
        a_write = start

        for k in range(m):
            tip = start + int(order[k])
            if used[tip] != 0 or dp_f[tip] < min_score:
                continue

            # Walk predecessor pointers to measure chain length and stopping point
            curr = tip
            cnt = 0
            stop_node = -1
            while curr >= start:
                if used[curr] != 0:
                    stop_node = curr
                    break
                cnt += 1
                curr = int(dp_p[curr])

            chain_sc = int(dp_f[tip]) - (int(dp_f[stop_node]) if stop_node >= start else 0)
            if cnt >= min_cnt and chain_sc >= min_score:
                # Commit chain in ascending (t_pos, q_pos) order
                curr = tip
                write_pos = a_write + cnt - 1
                while curr >= start and curr != stop_node:
                    used[curr] = 1
                    tmp_anchor_idx[write_pos] = curr
                    write_pos -= 1
                    curr = int(dp_p[curr])
                tmp_c_score[c_write] = chain_sc
                tmp_c_len[c_write] = cnt
                tmp_c_a_start[c_write] = a_write
                c_write += 1
                a_write += cnt

        pair_chain_counts[p_idx] = c_write - start
        pair_anchor_counts[p_idx] = a_write - start


@njit(parallel=True, cache=True, nogil=True)
def _compact_chains_kernel(
    q_indices: np.ndarray[tuple[int], np.dtype[np.uint32]],
    t_indices: np.ndarray[tuple[int], np.dtype[np.uint32]],
    q_pos: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_pos: np.ndarray[tuple[int], np.dtype[np.int32]],
    spans: np.ndarray[tuple[int], np.dtype[np.uint16]],
    pair_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    pair_chain_counts: np.ndarray[tuple[int], np.dtype[np.int32]],
    chain_indptr: np.ndarray[tuple[int], np.dtype[np.int32]],
    anchor_indptr: np.ndarray[tuple[int], np.dtype[np.int32]],
    tmp_c_score: np.ndarray[tuple[int], np.dtype[np.int32]],
    tmp_c_len: np.ndarray[tuple[int], np.dtype[np.int32]],
    tmp_c_a_start: np.ndarray[tuple[int], np.dtype[np.int32]],
    tmp_anchor_idx: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_q: np.ndarray[tuple[int], np.dtype[np.uint32]],
    out_t: np.ndarray[tuple[int], np.dtype[np.uint32]],
    out_scores: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_q_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_q_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_t_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_t_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_aq: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_at: np.ndarray[tuple[int], np.dtype[np.int32]],
    out_asp: np.ndarray[tuple[int], np.dtype[np.uint16]],
) -> None:
    r"""Compact per-pair chains and anchor slices into flat RaggedArrayContainer arrays."""
    n_pairs = len(pair_starts)
    for p_idx in prange(n_pairs):  # type: ignore[not-iterable]
        n_c = int(pair_chain_counts[p_idx])
        if n_c == 0:
            continue
        start = int(pair_starts[p_idx])
        q_id = q_indices[start]
        t_id = t_indices[start]
        c_out_base = int(chain_indptr[p_idx])
        a_out_curr = int(anchor_indptr[p_idx])

        for c in range(n_c):
            src_c = start + c
            cnt = int(tmp_c_len[src_c])
            src_a = int(tmp_c_a_start[src_c])
            dst_c = c_out_base + c

            first_a = int(tmp_anchor_idx[src_a])
            last_a = int(tmp_anchor_idx[src_a + cnt - 1])

            out_q[dst_c] = q_id
            out_t[dst_c] = t_id
            out_scores[dst_c] = tmp_c_score[src_c]
            out_q_starts[dst_c] = q_pos[first_a]
            out_q_ends[dst_c] = q_pos[last_a] + int(spans[last_a])
            out_t_starts[dst_c] = t_pos[first_a]
            out_t_ends[dst_c] = t_pos[last_a] + int(spans[last_a])
            out_offsets[dst_c] = a_out_curr
            out_lengths[dst_c] = cnt

            for k in range(cnt):
                orig_idx = int(tmp_anchor_idx[src_a + k])
                out_aq[a_out_curr + k] = q_pos[orig_idx]
                out_at[a_out_curr + k] = t_pos[orig_idx]
                out_asp[a_out_curr + k] = spans[orig_idx]

            a_out_curr += cnt


@njit(cache=True, nogil=True)
def _classify_primary_chains_kernel(
    chain_q: np.ndarray[tuple[int], np.dtype[np.uint32]],
    chain_scores: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
    mask_level: float,
) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    r"""Classify chains as primary vs secondary based on query interval overlap (`mask_level`)."""
    n = len(chain_q)
    is_primary = np.ones(n, dtype=np.bool_)
    if n <= 1:
        return is_primary

    run_start = 0
    while run_start < n:
        q_id = chain_q[run_start]
        run_end = run_start + 1
        while run_end < n and chain_q[run_end] == q_id:
            run_end += 1

        if run_end - run_start > 1:
            order = np.argsort(-chain_scores[run_start:run_end])
            for a in range(len(order)):
                idx_a = run_start + int(order[a])
                qs_a = int(q_starts[idx_a])
                qe_a = int(q_ends[idx_a])
                len_a = max(1, qe_a - qs_a)
                prim = True
                for b in range(a):
                    idx_b = run_start + int(order[b])
                    if not is_primary[idx_b]:
                        continue
                    qs_b = int(q_starts[idx_b])
                    qe_b = int(q_ends[idx_b])
                    ovl = min(qe_a, qe_b) - max(qs_a, qs_b)
                    if ovl > 0:
                        len_b = max(1, qe_b - qs_b)
                        min_len = len_a if len_a < len_b else len_b
                        if float(ovl) >= mask_level * float(min_len):
                            prim = False
                            break
                is_primary[idx_a] = prim

        run_start = run_end

    return is_primary
