r"""Alignment seed and chaining anchor Structure-of-Arrays (SoA) batch containers.

This module provides [`Seed`][seroverse.core.align.seeds.Seed] and [`Seeds`][seroverse.core.align.seeds.Seeds]
for representing both pair-level consensus diagonal seeds and unaggregated multi-anchor hits,
along with Numba-accelerated local ungapped diagonal refinement.
"""

from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from seroverse.core.align.score import SubstitutionMatrix, _score_ungapped_diagonal
from seroverse.core.collections import BatchedContainer, Int32Array, ScalarView, UInt16Array, UInt32Array
from seroverse.core.seq.seq import Sequences


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Seed(ScalarView):
    r"""Alignment seed / anchor representing a matching region between query and target.

    Attributes:
        query_index (int): Index of the query sequence.
        target_index (int): Index of the target sequence.
        score (int): Match score (shared k-mer count when aggregated, or anchor weight `k` when unaggregated).
        offset (int): Diagonal offset calculated as `q_pos - t_pos`.
        q_pos (int): 0-based start coordinate on the query sequence. Defaults to `0`.
        t_pos (int): 0-based start coordinate on the target sequence. Defaults to `0`.
        span (int): Anchor match span in residues/bases. Defaults to `1`.
    """

    query_index: int
    target_index: int
    score: int
    offset: int
    q_pos: int = 0
    t_pos: int = 0
    span: int = 1


@dataclass(frozen=True, slots=True)
class Seeds(BatchedContainer[Seed, "Seeds"]):
    r"""Structure-of-Arrays (SoA) batch container for alignment seeds and chaining anchors.

    Unifies aggregated pair-level seeds (1 consensus diagonal per `(query, target)` pair)
    and unaggregated multi-anchor hits (all `(q_pos, t_pos, span)` seed occurrences per pair)
    in a single container without redundancy.

    Attributes:
        query_indices (UInt32Array): 1D array of query sequence indices.
        target_indices (UInt32Array): 1D array of target sequence indices.
        scores (UInt32Array): 1D array of seed/anchor scores.
        offsets (Int32Array): 1D array of diagonal offsets (`q_pos - t_pos`).
        q_pos (Int32Array): 1D array of 0-based query start positions.
        t_pos (Int32Array): 1D array of 0-based target start positions.
        spans (UInt16Array): 1D array of anchor match lengths (`k`).
    """

    query_indices: UInt32Array
    target_indices: UInt32Array
    scores: UInt32Array
    offsets: Int32Array
    q_pos: Int32Array = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    t_pos: Int32Array = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    spans: UInt16Array = field(default_factory=lambda: np.empty(0, dtype=np.uint16))

    def __post_init__(self) -> None:
        r"""Populate default anchor coordinates if constructed with 4-argument legacy signature."""
        n = len(self.query_indices)
        if n > 0 and len(self.q_pos) == 0:
            object.__setattr__(self, "q_pos", np.maximum(0, self.offsets).astype(np.int32))
            object.__setattr__(self, "t_pos", np.maximum(0, -self.offsets).astype(np.int32))
        if n > 0 and len(self.spans) == 0:
            object.__setattr__(self, "spans", np.ones(n, dtype=np.uint16))

    def top_hits(self, min_score: int = 1) -> Seeds:
        r"""Reduce the batch to only the highest-scoring seed for each query.

        Args:
            min_score (int): Minimum match score threshold. Defaults to 1.

        Returns:
            Seeds: Filtered [`Seeds`][seroverse.core.align.seeds.Seeds] collection containing top hit per query.
        """
        if len(self) == 0:
            return self

        order = np.lexsort((~self.scores, self.query_indices))
        _, unique_idx = np.unique(self.query_indices[order], return_index=True)
        best_idx = order[unique_idx]
        best_idx.sort()

        best_batch = self[best_idx]
        if min_score > 0:
            return best_batch.filter(best_batch.scores >= min_score)

        return best_batch

    def extract_sequences(self, queries: Sequences, targets: Sequences) -> tuple[Sequences, Sequences]:
        r"""Extract parallel batches of query and target sequences mapped by this seed batch.

        Args:
            queries (Sequences): Source collection of query sequences.
            targets (Sequences): Source collection of target sequences.

        Returns:
            tuple[Sequences, Sequences]: Paired query and target sequence collections.
        """
        return queries[self.query_indices], targets[self.target_indices]

    def rescore_diagonals(
        self,
        queries: Sequences,
        targets: Sequences,
        matrix: SubstitutionMatrix = SubstitutionMatrix.BLOSUM62,
        window: int = 16,
    ) -> Seeds:
        r"""Refine diagonal `offsets` by scoring local ungapped homology within `[offset - window, offset + window]`.

        Args:
            queries (Sequences): Query sequences collection.
            targets (Sequences): Target sequences collection.
            matrix (SubstitutionMatrix): Substitution matrix for ungapped diagonal scoring.
                Defaults to `SubstitutionMatrix.BLOSUM62`.
            window (int): Half-window search radius around each seed's initial diagonal offset.
                Defaults to 16.

        Returns:
            Seeds: New [`Seeds`][seroverse.core.align.seeds.Seeds] batch with refined `offsets`.
        """
        if len(self) == 0 or window <= 0:
            return self

        refined_offsets = _rescore_seed_diagonals_kernel(
            self.query_indices,
            self.target_indices,
            self.offsets,
            queries.seqs,
            queries.offsets,
            queries.lengths,
            targets.seqs,
            targets.offsets,
            targets.lengths,
            matrix.lut,
            np.int32(window),
        )
        return Seeds(
            self.query_indices,
            self.target_indices,
            self.scores,
            refined_offsets,
            self.q_pos,
            self.t_pos,
            self.spans,
        )


# Numba Kernels --------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True, parallel=True)
def _rescore_seed_diagonals_kernel(
    query_indices: UInt32Array,
    target_indices: UInt32Array,
    seed_offsets: Int32Array,
    q_seqs: np.ndarray[tuple[int], np.dtype[np.uint8]],
    q_offsets: Int32Array,
    q_lengths: Int32Array,
    t_seqs: np.ndarray[tuple[int], np.dtype[np.uint8]],
    t_offsets: Int32Array,
    t_lengths: Int32Array,
    sub_matrix: np.ndarray[tuple[int, int], np.dtype[np.int8]],
    window: np.int32,
) -> Int32Array:
    r"""Refine seed diagonal offsets by scanning a local `[offset - window, offset + window]` diagonal band."""
    n = len(query_indices)
    out_offsets = np.empty(n, dtype=np.int32)

    for idx in prange(n):
        q_idx = query_indices[idx]
        t_idx = target_indices[idx]
        base_diag = seed_offsets[idx]

        q_off = q_offsets[q_idx]
        q_len = q_lengths[q_idx]
        t_off = t_offsets[t_idx]
        t_len = t_lengths[t_idx]

        best_diag = base_diag
        best_score = _score_ungapped_diagonal(q_seqs, q_off, q_len, t_seqs, t_off, t_len, base_diag, sub_matrix)

        min_d = max(-t_len + 1, base_diag - window)
        max_d = min(q_len - 1, base_diag + window)

        for d in range(min_d, max_d + 1):
            if d == base_diag:
                continue
            sc = _score_ungapped_diagonal(q_seqs, q_off, q_len, t_seqs, t_off, t_len, d, sub_matrix)
            if sc > best_score:
                best_score = sc
                best_diag = np.int32(d)

        out_offsets[idx] = best_diag

    return out_offsets
