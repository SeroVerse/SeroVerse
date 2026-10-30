r"""Compressed Sparse Row (`CSR`) and multi-relational CSR (`RelationalCSR`) graph matrices.

Provides zero-SciPy, Numba-accelerated sparse adjacency representations:
- [`CSR`][seroverse.core.graph.csr.CSR]: Base 2D Compressed Sparse Row matrix supporting fast
  COO/dense construction, transpose (`.T`), SpMV/SpMM (`@`), sparse addition (`+`), submatrix
  slicing, symmetric normalized Laplacian (`.laplacian`), and connected components
  (`.connected_components`).
- [`RelationalCSR`][seroverse.core.graph.csr.RelationalCSR]: Subclass of [`CSR`][seroverse.core.graph.csr.CSR]
  storing per-edge relation type IDs (`rel_types`) and virtual pool node indices (`virtual_nodes`)
  for multi-relational message passing.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, overload

import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    from seroverse.core.graph.models import Graph, Graphs


# ----------------------------------------------------------------------------------------------------------------------
# Compiled Numba CSR & Graph Kernels
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _compress_sorted_coo_kernel(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    n_rows: int,
    sum_duplicates: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compress lexicographically sorted `(rows, cols, data)` arrays into CSR `(indptr, indices, data)`."""
    nnz_in = len(rows)
    indptr = np.zeros(n_rows + 1, dtype=np.int32)
    if nnz_in == 0:
        return indptr, np.empty(0, dtype=np.int32), np.empty(0, dtype=data.dtype)

    if not sum_duplicates:
        indices = np.empty(nnz_in, dtype=np.int32)
        out_data = np.empty(nnz_in, dtype=data.dtype)
        for k in range(nnz_in):
            r = int(rows[k])
            indptr[r + 1] += 1
            indices[k] = int(cols[k])
            out_data[k] = data[k]
        for r in range(n_rows):
            indptr[r + 1] += indptr[r]
        return indptr, indices, out_data

    # Count unique (row, col) pairs first
    unique_count = 1
    prev_r = int(rows[0])
    prev_c = int(cols[0])
    for k in range(1, nnz_in):
        r = int(rows[k])
        c = int(cols[k])
        if r != prev_r or c != prev_c:
            unique_count += 1
            prev_r = r
            prev_c = c

    indices = np.empty(unique_count, dtype=np.int32)
    out_data = np.empty(unique_count, dtype=data.dtype)

    out_idx = 0
    prev_r = int(rows[0])
    prev_c = int(cols[0])
    acc = data[0]
    for k in range(1, nnz_in):
        r = int(rows[k])
        c = int(cols[k])
        if r == prev_r and c == prev_c:
            acc = acc + data[k]
        else:
            indices[out_idx] = prev_c
            out_data[out_idx] = acc
            indptr[prev_r + 1] += 1
            out_idx += 1
            prev_r = r
            prev_c = c
            acc = data[k]

    indices[out_idx] = prev_c
    out_data[out_idx] = acc
    indptr[prev_r + 1] += 1

    for r in range(n_rows):
        indptr[r + 1] += indptr[r]

    return indptr, indices, out_data


@njit(cache=True, nogil=True)
def _csr_transpose_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the transpose of a CSR matrix in $O(N + \\text{nnz})$ time."""
    nnz = len(indices)
    t_indptr = np.zeros(n_cols + 1, dtype=np.int32)
    for k in range(nnz):
        t_indptr[indices[k] + 1] += 1
    for c in range(n_cols):
        t_indptr[c + 1] += t_indptr[c]

    t_indices = np.empty(nnz, dtype=np.int32)
    t_data = np.empty(nnz, dtype=data.dtype)
    heads = t_indptr[:-1].copy()

    for r in range(n_rows):
        for k in range(indptr[r], indptr[r + 1]):
            c = indices[k]
            dest = heads[c]
            t_indices[dest] = r
            t_data[dest] = data[k]
            heads[c] = dest + 1

    return t_indptr, t_indices, t_data


@njit(cache=True, nogil=True, parallel=True)
def _csr_matmul_1d_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    vec: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Parallel CSR matrix-vector multiplication `y = A @ x`."""
    out = np.zeros(n_rows, dtype=np.float64)
    for r in prange(n_rows):
        acc = 0.0
        for k in range(indptr[r], indptr[r + 1]):
            acc += float(data[k]) * float(vec[indices[k]])
        out[r] = acc
    return out


@njit(cache=True, nogil=True, parallel=True)
def _csr_matmul_2d_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    mat: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Parallel CSR sparse-dense matrix multiplication `Y = A @ X`."""
    n_cols_out = mat.shape[1]
    out = np.zeros((n_rows, n_cols_out), dtype=np.float64)
    for r in prange(n_rows):
        for k in range(indptr[r], indptr[r + 1]):
            c = indices[k]
            w = float(data[k])
            for d in range(n_cols_out):
                out[r, d] += w * float(mat[c, d])
    return out


@njit(cache=True, nogil=True)
def _csr_row_sum_kernel(indptr: np.ndarray, data: np.ndarray, n_rows: int) -> np.ndarray:
    """Compute 1D row sums of a CSR matrix."""
    out = np.zeros(n_rows, dtype=np.float64)
    for r in range(n_rows):
        acc = 0.0
        for k in range(indptr[r], indptr[r + 1]):
            acc += float(data[k])
        out[r] = acc
    return out


@njit(cache=True, nogil=True)
def _csr_col_sum_kernel(indices: np.ndarray, data: np.ndarray, n_cols: int) -> np.ndarray:
    """Compute 1D column sums of a CSR matrix."""
    out = np.zeros(n_cols, dtype=np.float64)
    for k in range(len(indices)):
        out[indices[k]] += float(data[k])
    return out


@njit(cache=True, nogil=True)
def _csr_to_dense_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """Expand CSR arrays into a 2D dense array."""
    out = np.zeros((n_rows, n_cols), dtype=data.dtype)
    for r in range(n_rows):
        for k in range(indptr[r], indptr[r + 1]):
            out[r, indices[k]] = data[k]
    return out


@njit(cache=True, nogil=True)
def _csr_laplacian_dense_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_nodes: int,
    normed: bool,
    symmetrize_unweighted: bool,
) -> np.ndarray:
    """Compute a dense (optionally normalized and symmetrized) graph Laplacian directly from CSR arrays."""
    lap = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if n_nodes == 0:
        return lap

    for u in range(n_nodes):
        for k in range(indptr[u], indptr[u + 1]):
            v = int(indices[k])
            if u == v:
                continue
            w = float(data[k])
            if w == 0.0:
                continue
            if symmetrize_unweighted:
                lap[u, v] = 1.0
                lap[v, u] = 1.0
            else:
                lap[u, v] += w

    deg = np.zeros(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        s = 0.0
        for j in range(n_nodes):
            s += lap[i, j]
        deg[i] = s

    if normed:
        inv_sqrt_deg = np.zeros(n_nodes, dtype=np.float64)
        for i in range(n_nodes):
            if deg[i] > 0.0:
                inv_sqrt_deg[i] = 1.0 / np.sqrt(deg[i])
        for i in range(n_nodes):
            di = inv_sqrt_deg[i]
            for j in range(n_nodes):
                if i == j:
                    lap[i, i] = 1.0 if deg[i] > 0.0 else 0.0
                elif lap[i, j] != 0.0:
                    lap[i, j] = -lap[i, j] * di * inv_sqrt_deg[j]
    else:
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    lap[i, i] = deg[i]
                elif lap[i, j] != 0.0:
                    lap[i, j] = -lap[i, j]

    return lap


@njit(cache=True, nogil=True, inline="always")
def _dsu_union(parent: np.ndarray, u: int, v: int) -> None:
    """Union two vertices in a Disjoint Set Union array with path halving."""
    if u == v:
        return
    ru = u
    rv = v
    while parent[ru] != ru:
        parent[ru] = parent[parent[ru]]
        ru = int(parent[ru])
    while parent[rv] != rv:
        parent[rv] = parent[parent[rv]]
        rv = int(parent[rv])
    if ru != rv:
        if ru < rv:
            parent[rv] = ru
        else:
            parent[ru] = rv


@njit(cache=True, nogil=True, inline="always")
def _dsu_finalize(parent: np.ndarray) -> tuple[int, np.ndarray]:
    """Compress DSU parent pointers into contiguous `0..n_comp-1` component labels."""
    n_nodes = len(parent)
    labels = np.full(n_nodes, -1, dtype=np.int32)
    n_comp = 0
    for i in range(n_nodes):
        r = i
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = int(parent[r])
        if labels[r] == -1:
            labels[r] = n_comp
            n_comp += 1
        labels[i] = labels[r]
    return n_comp, labels


@njit(cache=True, nogil=True)
def _edges_connected_components_kernel(
    n_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Compute undirected connected components in $O(N + E)$ time over flat edge arrays."""
    if n_nodes <= 0:
        return 0, np.empty(0, dtype=np.int32)
    parent = np.arange(n_nodes, dtype=np.int32)
    for e in range(len(edge_u)):
        _dsu_union(parent, int(edge_u[e]), int(edge_v[e]))
    return _dsu_finalize(parent)


@njit(cache=True, nogil=True)
def _connected_components_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_nodes: int,
) -> tuple[int, np.ndarray]:
    """Compute undirected connected components in $O(N + \\text{nnz})$ time directly over CSR structure."""
    if n_nodes <= 0:
        return 0, np.empty(0, dtype=np.int32)
    parent = np.arange(n_nodes, dtype=np.int32)
    for u in range(n_nodes):
        for k in range(indptr[u], indptr[u + 1]):
            if data[k] != 0:
                _dsu_union(parent, u, int(indices[k]))
    return _dsu_finalize(parent)


@njit(cache=True, nogil=True)
def _extract_upper_undirected_edges_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_nodes: int,
    use_weights: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetrize a CSR matrix (`A + A.T`) and extract upper-triangular undirected edges `(u < v, w)`."""
    if n_nodes <= 1 or len(indices) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

    # First collect all directed non-self edges flipped to (min(u, v), max(u, v), w)
    max_e = len(indices)
    raw_u = np.empty(max_e, dtype=np.int32)
    raw_v = np.empty(max_e, dtype=np.int32)
    raw_w = np.empty(max_e, dtype=np.float64)
    m = 0
    for u in range(n_nodes):
        for k in range(indptr[u], indptr[u + 1]):
            v = int(indices[k])
            w = 1.0 if not use_weights else float(data[k])
            if u == v or w <= 0.0:
                continue
            if u < v:
                raw_u[m] = u
                raw_v[m] = v
            else:
                raw_u[m] = v
                raw_v[m] = u
            raw_w[m] = w
            m += 1

    if m == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

    # Sort by (u, v) via 64-bit composite key
    keys = np.empty(m, dtype=np.int64)
    for i in range(m):
        keys[i] = (np.int64(raw_u[i]) << 32) | np.int64(raw_v[i])
    order = np.argsort(keys)

    # Deduplicate and sum weights (or set to 1.0 when unweighted)
    uniq_u = np.empty(m, dtype=np.int32)
    uniq_v = np.empty(m, dtype=np.int32)
    uniq_w = np.empty(m, dtype=np.float64)

    idx0 = order[0]
    prev_k = keys[idx0]
    uniq_u[0] = raw_u[idx0]
    uniq_v[0] = raw_v[idx0]
    uniq_w[0] = 1.0 if not use_weights else raw_w[idx0]
    out_cnt = 1

    for i in range(1, m):
        oi = order[i]
        k_val = keys[oi]
        if k_val == prev_k:
            if use_weights:
                uniq_w[out_cnt - 1] += raw_w[oi]
        else:
            prev_k = k_val
            uniq_u[out_cnt] = raw_u[oi]
            uniq_v[out_cnt] = raw_v[oi]
            uniq_w[out_cnt] = 1.0 if not use_weights else raw_w[oi]
            out_cnt += 1

    return uniq_u[:out_cnt].copy(), uniq_v[:out_cnt].copy(), uniq_w[:out_cnt].copy()


@njit(cache=True, nogil=True)
def _build_2hop_edges_kernel(
    n_real: int,
    e_src: np.ndarray,
    e_dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 2-hop directed paths `u -> w -> v` (`u != v`) in CSR without Python lists."""
    n_e = len(e_src)
    if n_real == 0 or n_e == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    out_deg = np.zeros(n_real + 1, dtype=np.int32)
    for k in range(n_e):
        u = int(e_src[k])
        if 0 <= u < n_real:
            out_deg[u + 1] += 1
    for u in range(n_real):
        out_deg[u + 1] += out_deg[u]

    adj_dst = np.empty(n_e, dtype=np.int32)
    heads = out_deg[:-1].copy()
    for k in range(n_e):
        u = int(e_src[k])
        if 0 <= u < n_real:
            adj_dst[heads[u]] = int(e_dst[k])
            heads[u] += 1

    # Count 2-hop edges
    total_2hop = 0
    for u in range(n_real):
        for k1 in range(out_deg[u], out_deg[u + 1]):
            w = adj_dst[k1]
            if 0 <= w < n_real:
                for k2 in range(out_deg[w], out_deg[w + 1]):
                    v = adj_dst[k2]
                    if v != u:
                        total_2hop += 1

    if total_2hop == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    hop2_u = np.empty(total_2hop, dtype=np.int32)
    hop2_v = np.empty(total_2hop, dtype=np.int32)
    pos = 0
    for u in range(n_real):
        for k1 in range(out_deg[u], out_deg[u + 1]):
            w = adj_dst[k1]
            if 0 <= w < n_real:
                for k2 in range(out_deg[w], out_deg[w + 1]):
                    v = adj_dst[k2]
                    if v != u:
                        hop2_u[pos] = u
                        hop2_v[pos] = v
                        pos += 1

    return hop2_u, hop2_v


# ----------------------------------------------------------------------------------------------------------------------
# Base CSR Class
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class CSR:
    r"""Immutable 2D Compressed Sparse Row (`CSR`) matrix backed by contiguous NumPy arrays and Numba kernels.

    Supports direct attribute access (`indptr`, `indices`, `data`, `shape`) and standard sparse
    matrix operations (`@`, `+`, `.T`, `.sum()`, `.toarray()`, `.tocoo()`, `.laplacian()`,
    `.connected_components()`, `.with_diag()`, `.without_zeros()`, submatrix slicing) without
    requiring `scipy.sparse`.

    Attributes:
        indptr: 1D `int32` row pointer array of shape `(n_rows + 1,)`.
        indices: 1D `int32` column index array of shape `(nnz,)`.
        data: 1D non-zero value array of shape `(nnz,)`.
        shape: 2-tuple `(n_rows, n_cols)` giving matrix dimensions.
    """

    indptr: np.ndarray[tuple[int], np.dtype[np.int32]]
    indices: np.ndarray[tuple[int], np.dtype[np.int32]]
    data: np.ndarray[tuple[int], Any]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        r"""Normalize CSR arrays to contiguous `int32` index arrays and validated 2-tuple `shape`."""
        object.__setattr__(self, "indptr", np.ascontiguousarray(self.indptr, dtype=np.int32))
        object.__setattr__(self, "indices", np.ascontiguousarray(self.indices, dtype=np.int32))
        object.__setattr__(self, "data", np.ascontiguousarray(self.data))
        object.__setattr__(self, "shape", (int(self.shape[0]), int(self.shape[1])))

    # Classmethod Constructors -----------------------------------------------------------------------------------------
    @classmethod
    def empty(cls, shape: tuple[int, int], dtype: Any = np.float64) -> Self:
        r"""Create an empty `CSR` matrix of shape `(n_rows, n_cols)` with zero non-zeros."""
        nr, nc = int(shape[0]), int(shape[1])
        return cls(
            indptr=np.zeros(nr + 1, dtype=np.int32),
            indices=np.empty(0, dtype=np.int32),
            data=np.empty(0, dtype=dtype),
            shape=(nr, nc),
        )

    @classmethod
    def from_coo(
        cls,
        rows: Sequence[int] | np.ndarray,
        cols: Sequence[int] | np.ndarray,
        data: Sequence[Any] | np.ndarray | None = None,
        shape: tuple[int, int] | None = None,
        dtype: Any = None,
        *,
        sum_duplicates: bool = True,
    ) -> Self:
        r"""Construct a `CSR` matrix from coordinate `(rows, cols, data)` arrays."""
        r_arr = np.asarray(rows, dtype=np.int32)
        c_arr = np.asarray(cols, dtype=np.int32)
        if len(r_arr) != len(c_arr):
            raise ValueError(f"rows length ({len(r_arr)}) != cols length ({len(c_arr)}).")

        if data is None:
            dt = np.float64 if dtype is None else dtype
            d_arr = np.ones(len(r_arr), dtype=dt)
        else:
            d_arr = np.asarray(data, dtype=dtype) if dtype is not None else np.asarray(data)
            if len(d_arr) != len(r_arr):
                raise ValueError(f"data length ({len(d_arr)}) != rows length ({len(r_arr)}).")

        if shape is None:
            nr = int(np.max(r_arr)) + 1 if len(r_arr) > 0 else 0
            nc = int(np.max(c_arr)) + 1 if len(c_arr) > 0 else 0
        else:
            nr, nc = int(shape[0]), int(shape[1])

        if len(r_arr) == 0 or nr == 0 or nc == 0:
            return cls.empty((nr, nc), dtype=d_arr.dtype)

        keys = (r_arr.astype(np.int64) << 32) | (c_arr.astype(np.int64) & 0xFFFFFFFF)
        order = np.argsort(keys, kind="mergesort")
        r_sorted = np.ascontiguousarray(r_arr[order], dtype=np.int32)
        c_sorted = np.ascontiguousarray(c_arr[order], dtype=np.int32)
        d_sorted = np.ascontiguousarray(d_arr[order])

        indptr, indices, out_data = _compress_sorted_coo_kernel(
            r_sorted,
            c_sorted,
            d_sorted,
            nr,
            sum_duplicates,
        )
        return cls(indptr=indptr, indices=indices, data=out_data, shape=(nr, nc))

    @classmethod
    def from_dense(cls, arr: np.ndarray) -> Self:
        r"""Construct a `CSR` matrix from a 2D dense NumPy array."""
        mat = np.asarray(arr)
        if mat.ndim != 2:
            raise ValueError(f"Expected 2D array for CSR.from_dense, got shape {mat.shape}.")
        rows, cols = np.nonzero(mat)
        if len(rows) == 0:
            return cls.empty((int(mat.shape[0]), int(mat.shape[1])), dtype=mat.dtype)
        vals = mat[rows, cols]
        return cls.from_coo(rows, cols, vals, shape=(int(mat.shape[0]), int(mat.shape[1])), sum_duplicates=False)

    # Properties -------------------------------------------------------------------------------------------------------
    @property
    def ndim(self) -> int:
        r"""Number of matrix dimensions (always `2`)."""
        return 2

    @property
    def nnz(self) -> int:
        r"""Total number of stored elements."""
        return len(self.data)

    @property
    def dtype(self) -> np.dtype[Any]:
        r"""Data type of stored matrix values."""
        return self.data.dtype

    @property
    def row(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""1D `int32` array of row indices for each stored entry (COO compatibility)."""
        if self.nnz == 0:
            return np.empty(0, dtype=np.int32)
        return np.repeat(np.arange(self.shape[0], dtype=np.int32), np.diff(self.indptr))

    @property
    def col(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""1D `int32` array of column indices for each stored entry (COO compatibility)."""
        return self.indices

    @property
    def T(self) -> CSR:
        r"""Return the transpose $(N_{\text{cols}} \times N_{\text{rows}})$ as a `CSR` matrix."""
        nr, nc = self.shape
        if self.nnz == 0:
            return CSR.empty((nc, nr), dtype=self.dtype)
        t_indptr, t_indices, t_data = _csr_transpose_kernel(self.indptr, self.indices, self.data, nr, nc)
        return CSR(indptr=t_indptr, indices=t_indices, data=t_data, shape=(nc, nr))

    # Format & Copy Conversions ----------------------------------------------------------------------------------------
    def copy(self) -> Self:
        r"""Return a deep copy of this `CSR` matrix."""
        return self.__class__(
            indptr=self.indptr.copy(),
            indices=self.indices.copy(),
            data=self.data.copy(),
            shape=self.shape,
        )

    def astype(self, dtype: Any, copy: bool = True) -> CSR:
        r"""Cast `self.data` to `dtype` and return a `CSR` matrix."""
        if not copy and self.data.dtype == np.dtype(dtype):
            return self
        return CSR(
            indptr=self.indptr.copy() if copy else self.indptr,
            indices=self.indices.copy() if copy else self.indices,
            data=self.data.astype(dtype, copy=copy),
            shape=self.shape,
        )

    def tocsr(self) -> CSR:
        r"""Return `self` (compatibility with `scipy.sparse`)."""
        return self

    def tocsc(self) -> CSR:
        r"""Return `self.T` whose `.indptr` indexes columns (compatibility with `scipy.sparse`)."""
        return self.T

    def tocoo(self) -> CSR:
        r"""Return `self`, which exposes `.row`, `.col`, and `.data` directly."""
        return self

    def toarray(self) -> np.ndarray:
        r"""Convert this sparse `CSR` matrix into a 2D dense NumPy array."""
        nr, nc = self.shape
        if nr == 0 or nc == 0:
            return np.zeros((nr, nc), dtype=self.dtype)
        return _csr_to_dense_kernel(self.indptr, self.indices, self.data, nr, nc)

    # Functional Structural Transformations ----------------------------------------------------------------------------
    def with_diag(self, values: float = 0) -> CSR:
        r"""Return a new `CSR` matrix with diagonal entries `(i, i)` set to `values` (or removed when `values == 0`)."""
        if values == 0:
            if self.nnz == 0:
                return self
            rows = self.row
            keep = rows != self.indices
            if bool(np.all(keep)):
                return self
            return CSR.from_coo(rows[keep], self.indices[keep], self.data[keep], shape=self.shape, sum_duplicates=False)
        n_diag = min(self.shape[0], self.shape[1])
        diag_idx = np.arange(n_diag, dtype=np.int32)
        rows = self.row
        non_diag = rows != self.indices
        all_rows = np.concatenate([rows[non_diag], diag_idx])
        all_cols = np.concatenate([self.indices[non_diag], diag_idx])
        all_vals = np.concatenate([self.data[non_diag], np.full(n_diag, values, dtype=self.dtype)])
        return CSR.from_coo(all_rows, all_cols, all_vals, shape=self.shape, sum_duplicates=False)

    def without_zeros(self) -> CSR:
        r"""Return a new `CSR` matrix with zero-valued entries removed from `indices` and `data`."""
        if self.nnz == 0:
            return self
        nz = self.data != 0
        if bool(np.all(nz)):
            return self
        rows = self.row[nz]
        return CSR.from_coo(rows, self.indices[nz], self.data[nz], shape=self.shape, sum_duplicates=False)

    # Arithmetic & Reductions ------------------------------------------------------------------------------------------
    def sum(self, axis: int | None = None) -> np.ndarray | float | int:
        r"""Sum matrix elements over the given axis (`None`, `0`, or `1`)."""
        if axis is None:
            return self.data.sum()
        if axis == 1 or axis == -1:
            return _csr_row_sum_kernel(self.indptr, self.data, self.shape[0])
        if axis == 0 or axis == -2:
            return _csr_col_sum_kernel(self.indices, self.data, self.shape[1])
        raise ValueError(f"axis must be None, 0, or 1; got {axis}.")

    def __add__(self, other: CSR) -> CSR:
        r"""Sparse matrix addition `self + other`."""
        if not isinstance(other, CSR):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch in CSR addition: {self.shape} vs {other.shape}.")
        if self.nnz == 0:
            return other.copy()
        if other.nnz == 0:
            return self.copy()
        res_dtype = np.result_type(self.dtype, other.dtype)
        rows = np.concatenate([self.row, other.row])
        cols = np.concatenate([self.indices, other.indices])
        vals = np.concatenate([self.data.astype(res_dtype, copy=False), other.data.astype(res_dtype, copy=False)])
        return CSR.from_coo(rows, cols, vals, shape=self.shape, sum_duplicates=True)

    @overload
    def __matmul__(self, other: np.ndarray) -> np.ndarray: ...

    @overload
    def __matmul__(self, other: CSR) -> CSR: ...

    def __matmul__(self, other: np.ndarray | CSR) -> np.ndarray | CSR:
        r"""Multiply this `CSR` matrix by a 1D/2D dense NumPy array or another `CSR` matrix."""
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) != self.shape[1]:
                    raise ValueError(f"Dimension mismatch in CSR @ 1D array: {self.shape} vs {other.shape}.")
                return _csr_matmul_1d_kernel(self.indptr, self.indices, self.data, other, self.shape[0])
            if other.ndim == 2:
                if other.shape[0] != self.shape[1]:
                    raise ValueError(f"Dimension mismatch in CSR @ 2D array: {self.shape} vs {other.shape}.")
                return _csr_matmul_2d_kernel(self.indptr, self.indices, self.data, other, self.shape[0])
            raise ValueError(f"CSR @ ndarray only supports 1D or 2D arrays; got {other.ndim}D.")
        if isinstance(other, CSR):
            return CSR.from_dense(self.toarray() @ other.toarray())
        return NotImplemented

    def __getitem__(self, item: Any) -> Any:
        r"""Support scalar lookup `csr[i, j]` and 2D submatrix slicing `csr[rows, cols]`."""
        if not isinstance(item, tuple) or len(item) != 2:
            raise IndexError("CSR indexing requires a 2-tuple (row_idx, col_idx).")
        r_idx, c_idx = item
        nr, nc = self.shape

        if isinstance(r_idx, (int, np.integer)) and isinstance(c_idx, (int, np.integer)):
            r = int(r_idx)
            c = int(c_idx)
            if r < 0:
                r += nr
            if c < 0:
                c += nc
            if r < 0 or r >= nr or c < 0 or c >= nc:
                raise IndexError(f"CSR index ({r_idx}, {c_idx}) out of bounds for shape {self.shape}.")
            row_cols = self.indices[self.indptr[r] : self.indptr[r + 1]]
            row_vals = self.data[self.indptr[r] : self.indptr[r + 1]]
            matches = row_cols == c
            return row_vals[matches].sum() if np.any(matches) else self.dtype.type(0)

        def _resolve_axis(sel: Any, n: int) -> np.ndarray:
            if isinstance(sel, slice):
                return np.arange(*sel.indices(n), dtype=np.int32)
            arr = np.asarray(sel)
            if arr.dtype == np.bool_:
                return np.nonzero(arr)[0].astype(np.int32)
            return arr.astype(np.int32, copy=False).ravel()

        sel_r = _resolve_axis(r_idx, nr)
        sel_c = _resolve_axis(c_idx, nc)
        out_nr, out_nc = len(sel_r), len(sel_c)
        if out_nr == 0 or out_nc == 0 or self.nnz == 0:
            return CSR.empty((out_nr, out_nc), dtype=self.dtype)

        r_map = np.full(nr, -1, dtype=np.int32)
        r_map[sel_r] = np.arange(out_nr, dtype=np.int32)
        c_map = np.full(nc, -1, dtype=np.int32)
        c_map[sel_c] = np.arange(out_nc, dtype=np.int32)

        new_r = r_map[self.row]
        new_c = c_map[self.indices]
        keep = (new_r >= 0) & (new_c >= 0)
        if not np.any(keep):
            return CSR.empty((out_nr, out_nc), dtype=self.dtype)
        return CSR.from_coo(
            new_r[keep],
            new_c[keep],
            self.data[keep],
            shape=(out_nr, out_nc),
            sum_duplicates=True,
        )

    # Graph Algorithms on CSR ------------------------------------------------------------------------------------------
    def laplacian(
        self,
        *,
        normed: bool = True,
        symmetrize_unweighted: bool = False,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        r"""Compute the dense (normalized or unnormalized) graph Laplacian matrix $L$ in a single Numba pass."""
        if self.shape[0] != self.shape[1]:
            raise ValueError(f"Graph Laplacian requires a square CSR matrix, got shape {self.shape}.")
        return _csr_laplacian_dense_kernel(
            self.indptr,
            self.indices,
            self.data,
            self.shape[0],
            normed,
            symmetrize_unweighted,
        )

    def connected_components(self) -> tuple[int, np.ndarray[tuple[int], np.dtype[np.int32]]]:
        r"""Compute undirected connected components `(n_components, labels)` in $O(N + \text{nnz})$ time."""
        if self.shape[0] != self.shape[1]:
            raise ValueError(f"connected_components requires a square CSR matrix, got shape {self.shape}.")
        return _connected_components_kernel(self.indptr, self.indices, self.data, self.shape[0])

    @staticmethod
    def connected_components_from_edges(
        n_nodes: int,
        edge_u: np.ndarray,
        edge_v: np.ndarray,
    ) -> tuple[int, np.ndarray[tuple[int], np.dtype[np.int32]]]:
        r"""Compute undirected connected components `(n_components, labels)` directly from flat edge arrays in $O(N + E)$ time."""
        return _edges_connected_components_kernel(
            int(n_nodes),
            np.ascontiguousarray(edge_u, dtype=np.int32),
            np.ascontiguousarray(edge_v, dtype=np.int32),
        )

    def extract_undirected_edges(
        self,
        *,
        use_weights: bool = True,
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.float64]],
    ]:
        r"""Symmetrize (`A + A.T`) and extract unique upper-triangular undirected edges `(u < v, w)`."""
        return _extract_upper_undirected_edges_kernel(
            self.indptr,
            self.indices,
            self.data,
            self.shape[0],
            use_weights,
        )

    def __repr__(self) -> str:
        return f"CSR(shape={self.shape}, nnz={self.nnz}, dtype={self.dtype})"


# ----------------------------------------------------------------------------------------------------------------------
# Multi-Relational CSR Subclass
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class RelationalCSR(CSR):
    r"""Immutable Compressed Sparse Row (`CSR`) incoming-edge representation of a multi-relational directed graph.

    Inherits `indptr`, `indices` (incoming source vertex IDs), `data` (normalized `float32` edge
    aggregation weights), and `shape` from [`CSR`][seroverse.core.graph.csr.CSR], while adding
    per-edge relation type IDs (`rel_types`), total relation count (`n_relations`), and optional
    virtual pool node indices (`virtual_nodes`).

    For each target vertex $v \in [0, N-1]$, the slice `k in range(indptr[v], indptr[v + 1])`
    enumerates incoming directed edges $(u_k \xrightarrow{r_k} v)$ with source vertex
    `u_k = indices[k]`, relation ID `r_k = rel_types[k]`, and normalized aggregation weight `data[k]`.

    Attributes:
        indptr: 1D `int32` array of shape `(N + 1,)` indexing incoming edges per target vertex.
        indices: 1D `int32` array of shape `(E,)` of source vertex indices.
        data: 1D `float32` array of shape `(E,)` of per-edge aggregation weights.
        shape: 2-tuple `(N, N)` giving total node dimensions.
        rel_types: 1D `int32` array of shape `(E,)` of relation type IDs in `[0, R - 1]`.
        n_relations: Total number of distinct relation types $R$.
        virtual_nodes: 1D `int32` array of virtual global pool vertex indices (empty if none).
    """

    rel_types: np.ndarray[tuple[int], np.dtype[np.int32]] = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    n_relations: int = 1
    virtual_nodes: np.ndarray[tuple[int], np.dtype[np.int32]] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )

    def __post_init__(self) -> None:
        r"""Normalize multi-relational CSR arrays and validate dimensions."""
        super().__post_init__()
        object.__setattr__(self, "data", np.ascontiguousarray(self.data, dtype=np.float32))
        object.__setattr__(self, "rel_types", np.ascontiguousarray(self.rel_types, dtype=np.int32))
        object.__setattr__(self, "n_relations", max(int(self.n_relations), 1))
        object.__setattr__(self, "virtual_nodes", np.ascontiguousarray(self.virtual_nodes, dtype=np.int32))

    @property
    def n_nodes(self) -> int:
        r"""Total number of vertices $N$ (including any virtual pool vertices)."""
        return self.shape[0]

    @classmethod
    def from_edges(
        cls,
        n_nodes: int,
        sources: Sequence[int] | np.ndarray,
        targets: Sequence[int] | np.ndarray,
        relations: Sequence[int] | np.ndarray | None = None,
        *,
        n_relations: int | None = None,
        add_self_loops: bool = True,
        normalize_by_degree: bool = True,
        virtual_nodes: np.ndarray | None = None,
    ) -> Self:
        r"""Construct a [`RelationalCSR`][seroverse.core.graph.csr.RelationalCSR] from flat directed edge arrays.

        Args:
            n_nodes: Total number of vertices $N$.
            sources: 1D array-like of source vertex indices `u`.
            targets: 1D array-like of target vertex indices `v`.
            relations: Optional 1D array-like of integer relation IDs `r >= 0`. If `None`, all edges
                receive relation `1` (when `add_self_loops=True`, self-loops receive relation `0`).
            n_relations: Total number of relation types $R$. Inferred if `None`.
            add_self_loops: If `True` (default), prepends a self-loop `(v -> v, r=0)` for every vertex.
            normalize_by_degree: If `True` (default), normalizes incoming weights per `(target, relation)`
                group so each relation's incoming messages average cleanly.
            virtual_nodes: Optional 1D `int32` array of virtual global pool vertex indices.

        Returns:
            RelationalCSR: Incoming-edge multi-relational CSR structure.
        """
        src_arr = np.asarray(sources, dtype=np.int32)
        dst_arr = np.asarray(targets, dtype=np.int32)
        if len(src_arr) != len(dst_arr):
            raise ValueError(f"sources length ({len(src_arr)}) != targets length ({len(dst_arr)}).")

        if relations is None:
            rel_arr = np.full(len(src_arr), 1 if add_self_loops else 0, dtype=np.int32)
        else:
            rel_arr = np.asarray(relations, dtype=np.int32)
            if len(rel_arr) != len(src_arr):
                raise ValueError(f"relations length ({len(rel_arr)}) != sources length ({len(src_arr)}).")

        if add_self_loops and n_nodes > 0:
            self_idx = np.arange(n_nodes, dtype=np.int32)
            self_rel = np.zeros(n_nodes, dtype=np.int32)
            src_arr = np.concatenate([self_idx, src_arr])
            dst_arr = np.concatenate([self_idx, dst_arr])
            rel_arr = np.concatenate([self_rel, rel_arr])

        n_rel = (
            (int(np.max(rel_arr)) + 1 if len(rel_arr) > 0 else 1) if n_relations is None else max(int(n_relations), 1)
        )
        v_nodes = (
            np.asarray(virtual_nodes, dtype=np.int32) if virtual_nodes is not None else np.empty(0, dtype=np.int32)
        )

        if n_nodes == 0 or len(src_arr) == 0:
            return cls(
                indptr=np.zeros(n_nodes + 1, dtype=np.int32),
                indices=np.empty(0, dtype=np.int32),
                data=np.empty(0, dtype=np.float32),
                shape=(n_nodes, n_nodes),
                rel_types=np.empty(0, dtype=np.int32),
                n_relations=n_rel,
                virtual_nodes=v_nodes,
            )

        order = np.lexsort((src_arr, rel_arr, dst_arr))
        src_sorted = np.ascontiguousarray(src_arr[order], dtype=np.int32)
        dst_sorted = np.ascontiguousarray(dst_arr[order], dtype=np.int32)
        rel_sorted = np.ascontiguousarray(rel_arr[order], dtype=np.int32)

        counts = np.bincount(dst_sorted, minlength=n_nodes).astype(np.int32)
        indptr = np.zeros(n_nodes + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])

        weights = np.ones(len(src_sorted), dtype=np.float32)
        if normalize_by_degree:
            key = dst_sorted.astype(np.int64) * n_rel + rel_sorted.astype(np.int64)
            _, inv, key_counts = np.unique(key, return_inverse=True, return_counts=True)
            weights = (1.0 / np.maximum(key_counts[inv], 1)).astype(np.float32)

        return cls(
            indptr=indptr,
            indices=src_sorted,
            data=np.ascontiguousarray(weights, dtype=np.float32),
            shape=(n_nodes, n_nodes),
            rel_types=rel_sorted,
            n_relations=n_rel,
            virtual_nodes=v_nodes,
        )

    @classmethod
    def from_multihop(
        cls,
        node_offsets: Sequence[int] | np.ndarray,
        edge_sources: Sequence[int] | np.ndarray,
        edge_targets: Sequence[int] | np.ndarray,
        edge_orientations: Sequence[int] | np.ndarray | None = None,
        *,
        include_2hop: bool = True,
        add_virtual_pool: bool = True,
    ) -> Self:
        r"""Build an 8-relation multi-hop [`RelationalCSR`][seroverse.core.graph.csr.RelationalCSR] for a graph or ragged batch.

        Constructs up to 8 directed relation types:
            - `0`: Self-loop (`v -> v`)
            - `1`: 1-hop forward (`u -> v`, codirectional)
            - `2`: 1-hop reverse (`v -> u`, codirectional)
            - `3`: 1-hop convergent/divergent (`u <-> v` when `edge_orientations != 0`)
            - `4`: 2-hop forward (`u -> w -> v`)
            - `5`: 2-hop reverse (`v -> w -> u`)
            - `6`: Real vertex to per-graph virtual global pool (`u -> pool_g`)
            - `7`: Per-graph virtual global pool to real vertex (`pool_g -> u`)
        """
        offsets = np.asarray(node_offsets, dtype=np.int32)
        n_graphs = max(len(offsets) - 1, 0)
        n_real = int(offsets[-1]) if len(offsets) > 0 else 0
        e_src = np.asarray(edge_sources, dtype=np.int32)
        e_dst = np.asarray(edge_targets, dtype=np.int32)
        e_ori = (
            np.asarray(edge_orientations, dtype=np.int32)
            if edge_orientations is not None
            else np.zeros(len(e_src), dtype=np.int32)
        )

        src_list: list[np.ndarray] = []
        dst_list: list[np.ndarray] = []
        rel_list: list[np.ndarray] = []

        if len(e_src) > 0:
            codir = e_ori == 0
            if np.any(codir):
                u_c = e_src[codir]
                v_c = e_dst[codir]
                src_list.extend([u_c, v_c])
                dst_list.extend([v_c, u_c])
                rel_list.extend([np.full(len(u_c), 1, dtype=np.int32), np.full(len(u_c), 2, dtype=np.int32)])
            non_codir = ~codir
            if np.any(non_codir):
                u_nc = e_src[non_codir]
                v_nc = e_dst[non_codir]
                src_list.extend([u_nc, v_nc])
                dst_list.extend([v_nc, u_nc])
                rel_list.extend([np.full(len(u_nc), 3, dtype=np.int32), np.full(len(u_nc), 3, dtype=np.int32)])

            if include_2hop and n_real > 0:
                h2u, h2v = _build_2hop_edges_kernel(n_real, e_src, e_dst)
                if len(h2u) > 0:
                    src_list.extend([h2u, h2v])
                    dst_list.extend([h2v, h2u])
                    rel_list.extend([np.full(len(h2u), 4, dtype=np.int32), np.full(len(h2u), 5, dtype=np.int32)])

        if add_virtual_pool and n_graphs > 0:
            virtual_nodes = np.arange(n_real, n_real + n_graphs, dtype=np.int32)
            total_nodes = n_real + n_graphs
            if n_real > 0:
                graph_Lengths = np.diff(offsets).astype(np.int32)
                nodes_all = np.arange(n_real, dtype=np.int32)
                v_pool_ids = np.repeat(virtual_nodes, graph_Lengths)
                src_list.extend([nodes_all, v_pool_ids])
                dst_list.extend([v_pool_ids, nodes_all])
                rel_list.extend([np.full(n_real, 6, dtype=np.int32), np.full(n_real, 7, dtype=np.int32)])
        else:
            virtual_nodes = np.empty(0, dtype=np.int32)
            total_nodes = n_real

        all_src = np.concatenate(src_list) if src_list else np.empty(0, dtype=np.int32)
        all_dst = np.concatenate(dst_list) if dst_list else np.empty(0, dtype=np.int32)
        all_rel = np.concatenate(rel_list) if rel_list else np.empty(0, dtype=np.int32)

        return cls.from_edges(
            total_nodes,
            all_src,
            all_dst,
            all_rel,
            n_relations=8,
            add_self_loops=True,
            normalize_by_degree=True,
            virtual_nodes=virtual_nodes,
        )

    @classmethod
    def from_batch(
        cls,
        batch: Graphs[Any],
        edge_orientations: Sequence[int] | np.ndarray | None = None,
        *,
        include_2hop: bool = True,
        add_virtual_pool: bool = True,
    ) -> Self:
        r"""Build a multi-hop [`RelationalCSR`][seroverse.core.graph.csr.RelationalCSR] directly from a [`Graphs`][seroverse.core.graph.models.Graphs] batch."""
        return cls.from_multihop(
            batch.node_offsets,
            batch.edge_sources,
            batch.edge_targets,
            edge_orientations=edge_orientations,
            include_2hop=include_2hop,
            add_virtual_pool=add_virtual_pool,
        )

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        *,
        include_2hop: bool = True,
        add_virtual_pool: bool = True,
    ) -> Self:
        r"""Build a multi-hop [`RelationalCSR`][seroverse.core.graph.csr.RelationalCSR] from a single [`Graph`][seroverse.core.graph.models.Graph]."""
        n_nodes = len(graph.nodes)
        edges = graph.edges
        n_edges = len(edges)
        sources = np.fromiter((e.source for e in edges), dtype=np.int32, count=n_edges)
        targets = np.fromiter((e.target for e in edges), dtype=np.int32, count=n_edges)
        return cls.from_multihop(
            np.array([0, n_nodes], dtype=np.int32),
            sources,
            targets,
            include_2hop=include_2hop,
            add_virtual_pool=add_virtual_pool,
        )

    def __repr__(self) -> str:
        return (
            f"RelationalCSR(n_nodes={self.n_nodes}, n_relations={self.n_relations}, "
            f"nnz={self.nnz}, virtual_nodes={len(self.virtual_nodes)})"
        )
