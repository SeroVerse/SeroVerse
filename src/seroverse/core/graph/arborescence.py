r"""Directed spanning arborescence solvers and Structure-of-Arrays (SoA) containers.

Provides compiled Numba branch-and-bound algorithms encapsulated in an object-oriented solver
and ragged SoA container hierarchy:

- [`Arborescence`][seroverse.core.graph.arborescence.Arborescence]: Scalar view of a single directed spanning arborescence.
- [`Arborescences`][seroverse.core.graph.arborescence.Arborescences]: Ragged Structure-of-Arrays container of arborescences.
- [`ArborescenceSolver`][seroverse.core.graph.arborescence.ArborescenceSolver]: Configurable solver for maximum/minimum and $k$-best directed spanning arborescences.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self, overload

import numpy as np
from numba import njit

from seroverse.core.collections import RaggedArrayContainer, ScalarView

_NEG_INF = -1e30


# ----------------------------------------------------------------------------------------------------------------------
# Private Numba Branch-and-Bound Kernels
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _prepare_sorted_incoming_candidates(
    weights_3d: np.ndarray,
    groups_3d: np.ndarray,
    active_mask: np.ndarray,
    root: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-sort valid incoming candidate edges `(u, slot, weight, group)` per target vertex `v`."""
    n, _, s_dim = weights_3d.shape
    max_cands = n * s_dim
    cand_u = np.full((n, max_cands), -1, dtype=np.int32)
    cand_s = np.full((n, max_cands), -1, dtype=np.int32)
    cand_g = np.full((n, max_cands), -1, dtype=np.int32)
    cand_w = np.full((n, max_cands), _NEG_INF, dtype=np.float64)
    cand_counts = np.zeros(n, dtype=np.int32)

    active_non_root = np.empty(n, dtype=np.int32)
    regrets = np.empty(n, dtype=np.float64)
    m_active = 0

    for v in range(n):
        if v == root or active_mask[v] == 0:
            continue
        cnt = 0
        for u in range(n):
            if u == v or active_mask[u] == 0:
                continue
            for s in range(s_dim):
                w = float(weights_3d[u, v, s])
                if w > _NEG_INF * 0.1:
                    cand_u[v, cnt] = u
                    cand_s[v, cnt] = s
                    cand_g[v, cnt] = int(groups_3d[u, v, s])
                    cand_w[v, cnt] = w
                    cnt += 1
        if cnt > 1:
            order = np.argsort(-cand_w[v, :cnt])
            cand_u[v, :cnt] = cand_u[v, :cnt][order]
            cand_s[v, :cnt] = cand_s[v, :cnt][order]
            cand_g[v, :cnt] = cand_g[v, :cnt][order]
            cand_w[v, :cnt] = cand_w[v, :cnt][order]
            regrets[m_active] = cand_w[v, 0] - cand_w[v, 1]
        elif cnt == 1:
            regrets[m_active] = 1e12
        else:
            regrets[m_active] = 1e15

        cand_counts[v] = cnt
        active_non_root[m_active] = v
        m_active += 1

    order_v = active_non_root[:m_active][np.argsort(-regrets[:m_active])]
    return cand_u, cand_s, cand_g, cand_w, cand_counts, order_v


@njit(cache=True, nogil=True)
def _branch_and_bound_k_best(
    weights_3d: np.ndarray,
    groups_3d: np.ndarray,
    active_mask: np.ndarray,
    root: int,
    k_best: int,
    max_backtracks: int = 250_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Iterative stack-based Numba branch-and-bound for exact constrained top-`k` arborescences."""
    n = weights_3d.shape[0]
    cand_u, cand_s, cand_g, cand_w, cand_counts, order_v = _prepare_sorted_incoming_candidates(
        weights_3d, groups_3d, active_mask, root
    )
    m = len(order_v)
    out_parents = np.full((k_best, n), -1, dtype=np.int32)
    out_slots = np.full((k_best, n), -1, dtype=np.int32)
    out_scores = np.full(k_best, _NEG_INF, dtype=np.float64)

    if m == 0:
        out_scores[0] = 0.0
        return out_parents, out_slots, out_scores, 1

    for idx in range(m):
        if cand_counts[order_v[idx]] == 0:
            return out_parents, out_slots, out_scores, 0

    suffix_bound = np.zeros(m + 1, dtype=np.float64)
    for idx in range(m - 1, -1, -1):
        suffix_bound[idx] = suffix_bound[idx + 1] + cand_w[order_v[idx], 0]

    used_groups = np.zeros(max(int(np.max(groups_3d)) + 1, 1), dtype=np.uint8)
    curr_parents = np.full(n, -1, dtype=np.int32)
    curr_slots = np.full(n, -1, dtype=np.int32)
    curr_cand_idx = np.zeros(m, dtype=np.int32)
    curr_score_prefix = np.zeros(m + 1, dtype=np.float64)

    depth = 0
    found = 0
    backtracks = 0

    while depth >= 0:
        advanced = False
        if depth == m:
            total_sc = curr_score_prefix[m]
            if found < k_best or total_sc > out_scores[k_best - 1]:
                ins = found if found < k_best else k_best - 1
                while ins > 0 and out_scores[ins - 1] < total_sc:
                    if ins < k_best:
                        out_scores[ins] = out_scores[ins - 1]
                        out_parents[ins] = out_parents[ins - 1]
                        out_slots[ins] = out_slots[ins - 1]
                    ins -= 1
                out_scores[ins] = total_sc
                out_parents[ins] = curr_parents.copy()
                out_slots[ins] = curr_slots.copy()
                if found < k_best:
                    found += 1
        else:
            v = int(order_v[depth])
            c_idx = int(curr_cand_idx[depth])
            n_cands = int(cand_counts[v])
            threshold = out_scores[k_best - 1] if found == k_best else _NEG_INF * 0.5

            while c_idx < n_cands:
                w = cand_w[v, c_idx]
                if curr_score_prefix[depth] + w + suffix_bound[depth + 1] <= threshold + 1e-12:
                    break

                g = int(cand_g[v, c_idx])
                if g >= 0 and used_groups[g] != 0:
                    c_idx += 1
                    continue

                u = int(cand_u[v, c_idx])
                curr = u
                steps = 0
                has_cycle = False
                while curr >= 0 and steps <= n:
                    if curr == v:
                        has_cycle = True
                        break
                    curr = int(curr_parents[curr])
                    steps += 1
                if has_cycle:
                    c_idx += 1
                    continue

                curr_cand_idx[depth] = c_idx
                curr_parents[v] = u
                curr_slots[v] = cand_s[v, c_idx]
                if g >= 0:
                    used_groups[g] = 1
                curr_score_prefix[depth + 1] = curr_score_prefix[depth] + w
                depth += 1
                if depth < m:
                    curr_cand_idx[depth] = 0
                advanced = True
                break

            if not advanced:
                backtracks += 1
                if backtracks >= max_backtracks and found > 0:
                    break

        if not advanced:
            depth -= 1
            if depth >= 0:
                v_prev = int(order_v[depth])
                c_prev = int(curr_cand_idx[depth])
                g_prev = int(cand_g[v_prev, c_prev])
                if g_prev >= 0:
                    used_groups[g_prev] = 0
                curr_parents[v_prev] = -1
                curr_slots[v_prev] = -1
                curr_cand_idx[depth] += 1

    return out_parents[:found], out_slots[:found], out_scores[:found], found


# ----------------------------------------------------------------------------------------------------------------------
# ScalarView & RaggedArrayContainer
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Arborescence(ScalarView):
    r"""Scalar view of a single directed spanning arborescence.

    Attributes:
        parents: 1D `int32` array of shape `(N,)` where `parents[v]` is the parent node $u$ of $v$
            (`-1` for `root` or excluded nodes).
        score: Total weight (or cost) of the selected arborescence edges.
        slots: 1D `int32` array of shape `(N,)` giving the selected channel/slot index $s$ on the
            incoming edge $(u \to v, s)$ (`0` for 2D inputs, `-1` for `root`).
        root: Root vertex index (`0 <= root < N`). Defaults to `0`.
    """

    parents: np.ndarray[tuple[int], np.dtype[np.int32]]
    score: float
    slots: np.ndarray[tuple[int], np.dtype[np.int32]]
    root: int = 0

    @property
    def n_nodes(self) -> int:
        r"""Total number of vertices $N$ in the parent array."""
        return len(self.parents)

    @property
    def edges(self) -> np.ndarray[tuple[int, int], np.dtype[np.int32]]:
        r"""2D `int32` array of shape `(M, 3)` containing `(parent_u, child_v, slot_s)` for each tree edge."""
        n = len(self.parents)
        if n <= 1:
            return np.empty((0, 3), dtype=np.int32)
        v_idx = np.arange(n, dtype=np.int32)
        valid = (v_idx != self.root) & (self.parents >= 0)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.int32)
        v_sel = v_idx[valid]
        return np.column_stack((self.parents[v_sel], v_sel, np.maximum(self.slots[v_sel], np.int32(0)))).astype(
            np.int32, copy=False
        )

    @classmethod
    def maximum(
        cls,
        weights: np.ndarray,
        root: int = 0,
        *,
        edge_groups: np.ndarray | None = None,
        required_nodes: np.ndarray | None = None,
        slot_exclusivity: bool = True,
        max_backtracks: int = 250_000,
    ) -> Self:
        r"""Find the maximum-weight directed spanning arborescence rooted at `root`."""
        res = ArborescenceSolver(
            mode="max", root=root, k=1, slot_exclusivity=slot_exclusivity, max_backtracks=max_backtracks
        ).solve(weights, root=root, edge_groups=edge_groups, required_nodes=required_nodes)
        return cls(parents=res.parents, score=res.score, slots=res.slots, root=res.root)

    @classmethod
    def minimum(
        cls,
        costs: np.ndarray,
        root: int = 0,
        *,
        edge_groups: np.ndarray | None = None,
        required_nodes: np.ndarray | None = None,
        slot_exclusivity: bool = True,
        max_backtracks: int = 250_000,
    ) -> Self:
        r"""Find the minimum-cost directed spanning arborescence rooted at `root`."""
        res = ArborescenceSolver(
            mode="min", root=root, k=1, slot_exclusivity=slot_exclusivity, max_backtracks=max_backtracks
        ).solve(costs, root=root, edge_groups=edge_groups, required_nodes=required_nodes)
        return cls(parents=res.parents, score=res.score, slots=res.slots, root=res.root)


@dataclass(slots=True, frozen=True)
class Arborescences(RaggedArrayContainer[Arborescence, "Arborescences"]):
    r"""Ragged Structure-of-Arrays (SoA) container of [`Arborescence`][seroverse.core.graph.arborescence.Arborescence] trees.

    Attributes:
        scores: 1D `float64` array of total tree scores/costs of shape `(B,)`.
        roots: 1D `int32` array of root vertex indices of shape `(B,)`.
        parents: Flat 1D `int32` array of concatenated parent pointers across all trees.
        slots: Flat 1D `int32` array of concatenated incoming edge slot indices across all trees.
        offsets: 1D `int32` array of starting indices into `parents` and `slots` per tree.
        lengths: 1D `int32` array of vertex counts per tree.
    """

    scores: np.ndarray[tuple[int], np.dtype[np.float64]]
    roots: np.ndarray[tuple[int], np.dtype[np.int32]]
    parents: np.ndarray[tuple[int], np.dtype[np.int32]]
    slots: np.ndarray[tuple[int], np.dtype[np.int32]]
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]]

    def __len__(self) -> int:
        r"""Return the number of arborescences in the container."""
        return len(self.offsets)

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> Arborescence: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Arborescences: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Arborescence | Arborescences: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Arborescence | Arborescences:
        r"""Extract a single [`Arborescence`][seroverse.core.graph.arborescence.Arborescence] or a sliced [`Arborescences`][seroverse.core.graph.arborescence.Arborescences] batch."""
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Arborescences index {item} out of bounds [0, {len(self) - 1}].")
            off = int(self.offsets[idx])
            length = int(self.lengths[idx])
            return Arborescence(
                parents=self.parents[off : off + length],
                score=float(self.scores[idx]),
                slots=self.slots[off : off + length],
                root=int(self.roots[idx]),
            )

        indices, new_offsets, new_lengths, (new_parents, new_slots) = self._slice_ragged(item, self.parents, self.slots)
        if len(indices) == 0:
            return self.empty()
        return type(self)(
            scores=self.scores[indices],
            roots=self.roots[indices],
            parents=new_parents,
            slots=new_slots,
            offsets=new_offsets,
            lengths=new_lengths,
        )

    @classmethod
    def empty(cls) -> Self:
        r"""Return an empty 0-length [`Arborescences`][seroverse.core.graph.arborescence.Arborescences] container."""
        return cls(
            scores=np.empty(0, dtype=np.float64),
            roots=np.empty(0, dtype=np.int32),
            parents=np.empty(0, dtype=np.int32),
            slots=np.empty(0, dtype=np.int32),
            offsets=np.empty(0, dtype=np.int32),
            lengths=np.empty(0, dtype=np.int32),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        r"""Concatenate multiple [`Arborescences`][seroverse.core.graph.arborescence.Arborescences] containers into a single batch."""
        non_empty = [b for b in batches if len(b) > 0]
        if not non_empty:
            return cls.empty()
        lengths = np.concatenate([b.lengths for b in non_empty])
        parents_parts = [b.parents for b in non_empty if len(b.parents) > 0]
        slots_parts = [b.slots for b in non_empty if len(b.slots) > 0]
        return cls(
            scores=np.concatenate([b.scores for b in non_empty]),
            roots=np.concatenate([b.roots for b in non_empty]),
            parents=np.concatenate(parents_parts) if parents_parts else np.empty(0, dtype=np.int32),
            slots=np.concatenate(slots_parts) if slots_parts else np.empty(0, dtype=np.int32),
            offsets=cls.lengths_to_offsets(lengths),
            lengths=lengths,
        )

    @classmethod
    def from_list(cls, items: Sequence[Arborescence]) -> Self:
        r"""Pack a sequence of [`Arborescence`][seroverse.core.graph.arborescence.Arborescence] scalar views into a ragged container."""
        if not items:
            return cls.empty()
        lengths = np.array([len(a.parents) for a in items], dtype=np.int32)
        has_elems = bool(np.any(lengths > 0))
        return cls(
            scores=np.array([a.score for a in items], dtype=np.float64),
            roots=np.array([a.root for a in items], dtype=np.int32),
            parents=np.concatenate([a.parents.astype(np.int32, copy=False) for a in items if len(a.parents) > 0])
            if has_elems
            else np.empty(0, dtype=np.int32),
            slots=np.concatenate([a.slots.astype(np.int32, copy=False) for a in items if len(a.slots) > 0])
            if has_elems
            else np.empty(0, dtype=np.int32),
            offsets=cls.lengths_to_offsets(lengths),
            lengths=lengths,
        )

    @classmethod
    def k_best(
        cls,
        weights: np.ndarray,
        root: int = 0,
        k: int = 5,
        *,
        mode: Literal["max", "min"] = "max",
        edge_groups: np.ndarray | None = None,
        required_nodes: np.ndarray | None = None,
        slot_exclusivity: bool = True,
        max_backtracks: int = 250_000,
    ) -> Self:
        r"""Enumerate the top-`k` optimal directed spanning arborescences rooted at `root`."""
        res = ArborescenceSolver(
            mode=mode, root=root, k=k, slot_exclusivity=slot_exclusivity, max_backtracks=max_backtracks
        ).solve_k_best(weights, root=root, k=k, edge_groups=edge_groups, required_nodes=required_nodes)
        return cls(
            scores=res.scores,
            roots=res.roots,
            parents=res.parents,
            slots=res.slots,
            offsets=res.offsets,
            lengths=res.lengths,
        )


# ----------------------------------------------------------------------------------------------------------------------
# Configurable Object-Oriented Arborescence Solver
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class ArborescenceSolver:
    r"""Configurable solver for directed spanning arborescences and $k$-best arborescence ensembles.

    Uses compiled Numba branch-and-bound with regret-ordered vertex branching and suffix upper-bound
    pruning. Supports 2D `(N, N)` and 3D `(N, N, S)` edge tensors, valence/slot exclusivity
    (`slot_exclusivity=True`), custom `edge_groups` constraints, and subset `required_nodes`.

    Attributes:
        mode: Optimization sense (`"max"` for maximum weight or `"min"` for minimum cost). Defaults to `"max"`.
        root: Default root vertex index (`0 <= root < N`). Defaults to `0`.
        k: Default number of top arborescences to enumerate (`k >= 1`). Defaults to `1`.
        slot_exclusivity: If `True` (default), enforces that at most one outgoing edge from parent $u$
            may occupy slot $s$ on 3D `(N, N, S)` weight tensors when `edge_groups` is `None`.
        max_backtracks: Maximum branch-and-bound backtracks before returning best found solutions. Defaults to `250_000`.
    """

    mode: Literal["max", "min"] = "max"
    root: int = 0
    k: int = 1
    slot_exclusivity: bool = True
    max_backtracks: int = 250_000

    def __post_init__(self) -> None:
        r"""Validate solver hyperparameters."""
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min'; got {self.mode!r}.")
        if self.k < 1:
            raise ValueError(f"k must be >= 1; got {self.k}.")
        if self.max_backtracks < 1:
            raise ValueError(f"max_backtracks must be >= 1; got {self.max_backtracks}.")

    def _prepare_inputs(
        self,
        weights: np.ndarray,
        root: int,
        edge_groups: np.ndarray | None,
        required_nodes: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize 2D `(N, N)` or 3D `(N, N, S)` weights, active vertex mask, and exclusivity groups."""
        w_arr = np.asarray(weights, dtype=np.float64)
        if self.mode == "min":
            w_arr = np.where(np.isinf(w_arr) & (w_arr > 0), _NEG_INF, -w_arr)

        if w_arr.ndim == 2:
            n, n2 = w_arr.shape
            if n != n2:
                raise ValueError(f"2D weights matrix must be square (N, N); got {w_arr.shape}.")
            w_3d = np.ascontiguousarray(w_arr[:, :, None])
            s_dim = 1
        elif w_arr.ndim == 3:
            n, n2, s_dim = w_arr.shape
            if n != n2:
                raise ValueError(f"3D weights tensor must have shape (N, N, S); got {w_arr.shape}.")
            w_3d = np.ascontiguousarray(w_arr)
        else:
            raise ValueError(f"weights must be 2D (N, N) or 3D (N, N, S); got {w_arr.ndim}D.")

        if n > 0 and (root < 0 or root >= n):
            raise ValueError(f"root index {root} out of range [0, {n - 1}].")

        active_mask = np.ones(n, dtype=np.uint8)
        if required_nodes is not None:
            req = np.asarray(required_nodes)
            if req.dtype == np.bool_:
                if len(req) != n:
                    raise ValueError(f"Boolean required_nodes mask length {len(req)} != {n}.")
                active_mask = req.astype(np.uint8)
            else:
                active_mask.fill(0)
                active_mask[req.astype(np.int32)] = 1
            if n > 0:
                active_mask[root] = 1

        if edge_groups is not None:
            g_arr = np.asarray(edge_groups, dtype=np.int32)
            if g_arr.ndim == 2 and w_arr.ndim == 2:
                g_3d = np.ascontiguousarray(g_arr[:, :, None])
            elif g_arr.shape == w_3d.shape:
                g_3d = np.ascontiguousarray(g_arr)
            else:
                raise ValueError(f"edge_groups shape {g_arr.shape} incompatible with weights shape {w_arr.shape}.")
        elif w_arr.ndim == 3 and self.slot_exclusivity and s_dim > 1:
            g_3d = np.broadcast_to(
                (np.arange(n, dtype=np.int32)[:, None, None] * s_dim) + np.arange(s_dim, dtype=np.int32)[None, None, :],
                (n, n, s_dim),
            ).copy()
        else:
            g_3d = np.full((n, n, s_dim), -1, dtype=np.int32)

        return w_3d, g_3d, active_mask

    def solve_k_best(
        self,
        weights: np.ndarray,
        *,
        root: int | None = None,
        k: int | None = None,
        edge_groups: np.ndarray | None = None,
        required_nodes: np.ndarray | None = None,
    ) -> Arborescences:
        r"""Enumerate the top-`k` optimal arborescences in a ragged [`Arborescences`][seroverse.core.graph.arborescence.Arborescences] container."""
        r_idx = self.root if root is None else int(root)
        k_eff = self.k if k is None else int(k)
        if k_eff < 1:
            raise ValueError(f"k must be >= 1; got {k_eff}.")

        w_3d, g_3d, active_mask = self._prepare_inputs(weights, r_idx, edge_groups, required_nodes)
        n = w_3d.shape[0]
        if n <= 1:
            return Arborescences(
                scores=np.zeros(1, dtype=np.float64),
                roots=np.zeros(1, dtype=np.int32),
                parents=np.full(n, -1, dtype=np.int32),
                slots=np.full(n, -1, dtype=np.int32),
                offsets=np.zeros(1, dtype=np.int32),
                lengths=np.array([n], dtype=np.int32),
            )

        out_p, out_s, out_sc, found = _branch_and_bound_k_best(
            w_3d, g_3d, active_mask, r_idx, k_eff, self.max_backtracks
        )
        if found == 0:
            return Arborescences.empty()

        scores = (
            np.where(out_sc > _NEG_INF * 0.1, -out_sc, np.inf).astype(np.float64)
            if self.mode == "min"
            else out_sc.astype(np.float64, copy=False)
        )
        lengths = np.full(found, n, dtype=np.int32)
        return Arborescences(
            scores=scores,
            roots=np.full(found, r_idx, dtype=np.int32),
            parents=np.ascontiguousarray(out_p.reshape(-1), dtype=np.int32),
            slots=np.ascontiguousarray(out_s.reshape(-1), dtype=np.int32),
            offsets=Arborescences.lengths_to_offsets(lengths),
            lengths=lengths,
        )

    def solve(
        self,
        weights: np.ndarray,
        *,
        root: int | None = None,
        edge_groups: np.ndarray | None = None,
        required_nodes: np.ndarray | None = None,
    ) -> Arborescence:
        r"""Solve for the single optimal [`Arborescence`][seroverse.core.graph.arborescence.Arborescence]."""
        r_idx = self.root if root is None else int(root)
        batch = self.solve_k_best(weights, root=r_idx, k=1, edge_groups=edge_groups, required_nodes=required_nodes)
        if len(batch) > 0:
            return batch[0]
        n = np.asarray(weights).shape[0]
        return Arborescence(
            parents=np.full(n, -1, dtype=np.int32),
            score=float("inf") if self.mode == "min" else float(_NEG_INF),
            slots=np.full(n, -1, dtype=np.int32),
            root=r_idx,
        )

    def __call__(
        self,
        weights: np.ndarray,
        *,
        root: int | None = None,
        edge_groups: np.ndarray | None = None,
        required_nodes: np.ndarray | None = None,
    ) -> Arborescence | Arborescences:
        r"""Run `.solve()` when `self.k == 1`, or `.solve_k_best()` when `self.k > 1`."""
        if self.k == 1:
            return self.solve(weights, root=root, edge_groups=edge_groups, required_nodes=required_nodes)
        return self.solve_k_best(weights, root=root, k=self.k, edge_groups=edge_groups, required_nodes=required_nodes)
