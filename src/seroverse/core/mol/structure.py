"""3D spatial conformers and macromolecular structure containers (`Conformer`, `Conformers`, `Structure`, `Structures`)."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import numpy as np
from numba import prange

from seroverse.core.collections import RaggedBatch, View
from seroverse.core.graph.csr import CSR
from seroverse.core.mol.elements import Element
from seroverse.core.mol.mol import Atom, Bond, BondEdge, BondOrder, Mol, Mols
from seroverse.core.types import Types

_EMPTY_F32: Types.Float32Array = np.empty(0, dtype=np.float32)
_EMPTY_F32.setflags(write=False)


# 3D Conformer View & Ragged Batch -------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Conformer(View):
    r"""Immutable 3D spatial atomic coordinate and crystallographic metadata view.

    Stores Structure-of-Arrays (SoA) 1D `float32` coordinate channels (`x`, `y`, `z` in Ångströms)
    together with per-atom crystallographic `b_factors` (isotropic temperature factors) and `occupancies`.
    """

    x: Types.Float32Array = field(default_factory=lambda: _EMPTY_F32)
    y: Types.Float32Array = field(default_factory=lambda: _EMPTY_F32)
    z: Types.Float32Array = field(default_factory=lambda: _EMPTY_F32)
    b_factors: Types.Float32Array = field(default_factory=lambda: _EMPTY_F32)
    occupancies: Types.Float32Array = field(default_factory=lambda: _EMPTY_F32)

    def __len__(self) -> int:
        return len(self.x)

    @classmethod
    def empty(cls) -> Self:
        r"""Return an empty 0-atom `Conformer`."""
        return cls()

    @classmethod
    def from_coords(
        cls,
        coords: np.ndarray[Any, Any] | Sequence[Sequence[float]],
        *,
        b_factors: np.ndarray[Any, Any] | Sequence[float] | None = None,
        occupancies: np.ndarray[Any, Any] | Sequence[float] | None = None,
    ) -> Self:
        r"""Construct a [`Conformer`][seroverse.core.mol.structure.Conformer] from an `(N, 3)` coordinate array."""
        arr = np.asarray(coords, dtype=np.float32)
        if arr.size == 0:
            return cls.empty()
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3), got {arr.shape}")
        n = arr.shape[0]
        x = np.ascontiguousarray(arr[:, 0], dtype=np.float32)
        y = np.ascontiguousarray(arr[:, 1], dtype=np.float32)
        z = np.ascontiguousarray(arr[:, 2], dtype=np.float32)
        bf = np.zeros(n, dtype=np.float32) if b_factors is None else np.ascontiguousarray(b_factors, dtype=np.float32)
        occ = (
            np.ones(n, dtype=np.float32) if occupancies is None else np.ascontiguousarray(occupancies, dtype=np.float32)
        )
        if len(bf) != n or len(occ) != n:
            raise ValueError(f"b_factors ({len(bf)}) and occupancies ({len(occ)}) must match coords length ({n})")
        return cls(x=x, y=y, z=z, b_factors=bf, occupancies=occ)

    @property
    def coords(self) -> Types.Float32Array2D:
        r"""Dense `(N, 3)` `float32` coordinate matrix in Ångströms."""
        if len(self.x) == 0:
            return np.empty((0, 3), dtype=np.float32)
        return np.column_stack((self.x, self.y, self.z))

    def centroid(self) -> tuple[float, float, float]:
        r"""Compute the geometric 3D centroid `(cx, cy, cz)` in Ångströms."""
        cx, cy, cz = _conformer_centroid_kernel(self.x, self.y, self.z)
        return float(cx), float(cy), float(cz)

    def radius_of_gyration(self) -> float:
        r"""Compute the unweighted 3D radius of gyration $R_g$ in Ångströms."""
        return float(_conformer_rg_kernel(self.x, self.y, self.z))

    def center(self) -> Self:
        r"""Return a new `Conformer` translated so its centroid lies at `(0, 0, 0)`."""
        if len(self.x) == 0:
            return self
        cx, cy, cz = _conformer_centroid_kernel(self.x, self.y, self.z)
        return self.replace(
            x=(self.x - np.float32(cx)).astype(np.float32),
            y=(self.y - np.float32(cy)).astype(np.float32),
            z=(self.z - np.float32(cz)).astype(np.float32),
        )

    def distance_matrix(self) -> Types.Float32Array2D:
        r"""Compute the pairwise `(N, N)` 3D Euclidean distance matrix in Ångströms."""
        return _pairwise_dist3d_kernel(self.x, self.y, self.z)

    def contact_csr(self, cutoff: float = 5.0, *, include_self: bool = False) -> CSR:
        r"""Build a sparse `(N, N)` [`CSR`][seroverse.core.graph.csr.CSR] spatial contact graph within `cutoff` Å.

        Non-zero edge values store the Euclidean distance (in `float32` Å) between contacting atoms.
        """
        n = len(self.x)
        if n == 0:
            return CSR.empty((0, 0), dtype=np.float32)
        rows, cols, dists = _contact_edges_kernel(self.x, self.y, self.z, float(cutoff), bool(include_self))
        if len(rows) == 0:
            return CSR.empty((n, n), dtype=np.float32)
        return CSR.from_coo(rows, cols, dists, shape=(n, n), dtype=np.float32)

    def rmsd_to(self, other: Conformer) -> float:
        r"""Compute the optimal Kabsch-superimposed Root-Mean-Square Deviation (RMSD, in Å) to `other`."""
        if len(self.x) != len(other.x):
            raise ValueError(f"Conformer atom counts must match for RMSD ({len(self.x)} != {len(other.x)})")
        if len(self.x) == 0:
            return 0.0
        rmsd, _, _, _ = _kabsch_align_kernel(self.x, self.y, self.z, other.x, other.y, other.z)
        return float(rmsd)

    def align_to(self, reference: Conformer) -> Self:
        r"""Return a new `Conformer` optimally rotated and translated onto `reference` via the Kabsch algorithm."""
        if len(self.x) != len(reference.x):
            raise ValueError(f"Conformer atom counts must match for alignment ({len(self.x)} != {len(reference.x)})")
        if len(self.x) == 0:
            return self
        _, ax, ay, az = _kabsch_align_kernel(self.x, self.y, self.z, reference.x, reference.y, reference.z)
        return self.replace(x=ax, y=ay, z=az)

    def infer_covalent_bonds(
        self,
        atoms: Sequence[Atom | Element | int],
        *,
        tolerance: float = 0.45,
    ) -> tuple[BondEdge, ...]:
        r"""Infer covalent single bonds from 3D interatomic distances and element covalent radii ($r_i + r_j + \text{tol}$)."""
        n = len(self.x)
        if n != len(atoms):
            raise ValueError(f"atoms length ({len(atoms)}) must match conformer length ({n})")
        if n < 2:
            return ()
        elem_ids = np.fromiter((int(a) & 0x7F for a in atoms), dtype=np.uint32, count=n)
        srcs, tgts = _infer_covalent_bonds_kernel(
            self.x,
            self.y,
            self.z,
            elem_ids,
            Element.specs().covalent_radii,
            float(tolerance),
        )
        single_bond = Bond(BondOrder.SINGLE)
        return tuple(
            BondEdge(source=int(srcs[k]), target=int(tgts[k]), edge_type=single_bond) for k in range(len(srcs))
        )


@dataclass(slots=True, frozen=True)
class Conformers(RaggedBatch[Conformer]):
    r"""Ragged Structure-of-Arrays (SoA) batch container for [`Conformer`][seroverse.core.mol.structure.Conformer] records."""

    _ragged_fields: ClassVar[tuple[str, ...]] = ("x", "y", "z", "b_factors", "occupancies")
    _scalar_type: ClassVar[type[Any] | None] = Conformer

    offsets: Types.Int32Array
    lengths: Types.Int32Array
    x: Types.Float32Array
    y: Types.Float32Array
    z: Types.Float32Array
    b_factors: Types.Float32Array
    occupancies: Types.Float32Array

    def centroids(self) -> Types.Float32Array2D:
        r"""Compute the `(len(self), 3)` geometric centroids across all conformers in parallel."""
        return _batch_centroids_kernel(self.x, self.y, self.z, self.offsets, self.lengths)

    def radii_of_gyration(self) -> Types.Float32Array:
        r"""Compute the 1D `(len(self),)` radii of gyration ($R_g$ in Å) across all conformers in parallel."""
        return _batch_rg_kernel(self.x, self.y, self.z, self.offsets, self.lengths)


# 3D Molecular Structure & SoA Container -------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Structure(Mol):
    r"""Unified 3D molecular or macromolecular structure combining a [`Mol`][seroverse.core.mol.mol.Mol] graph with a [`Conformer`][seroverse.core.mol.structure.Conformer]."""

    nodes: tuple[Atom, ...] = ()
    edges: tuple[BondEdge, ...] = ()
    canonical: bytes = b""
    conformer: Conformer = field(default_factory=Conformer.empty)

    @classmethod
    def from_coords(
        cls,
        atoms: Sequence[Atom | Element | str | bytes],
        coords: np.ndarray[Any, Any] | Sequence[Sequence[float]],
        *,
        edges: Sequence[BondEdge] | None = None,
        canonical: bytes = b"",
        infer_bonds: bool = True,
        tolerance: float = 0.45,
        b_factors: np.ndarray[Any, Any] | Sequence[float] | None = None,
        occupancies: np.ndarray[Any, Any] | Sequence[float] | None = None,
    ) -> Self:
        r"""Construct a 3D [`Structure`][seroverse.core.mol.structure.Structure] from atoms and 3D coordinates, optionally inferring covalent bonds."""
        typed_atoms = tuple(a if isinstance(a, Atom) else Atom(a) for a in atoms)
        conf = Conformer.from_coords(coords, b_factors=b_factors, occupancies=occupancies)
        if len(typed_atoms) != len(conf):
            raise ValueError(f"atoms length ({len(typed_atoms)}) must match coords length ({len(conf)})")
        if edges is not None:
            typed_edges = tuple(edges)
        elif infer_bonds and len(typed_atoms) >= 2:
            typed_edges = conf.infer_covalent_bonds(typed_atoms, tolerance=tolerance)
        else:
            typed_edges = ()
        return cls(nodes=typed_atoms, edges=typed_edges, canonical=canonical, conformer=conf)

    def centroid(self) -> tuple[float, float, float]:
        r"""Geometric 3D centroid `(cx, cy, cz)` of this structure in Ångströms."""
        return self.conformer.centroid()

    def radius_of_gyration(self) -> float:
        r"""Unweighted 3D radius of gyration $R_g$ of this structure in Ångströms."""
        return self.conformer.radius_of_gyration()

    def distance_matrix(self) -> Types.Float32Array2D:
        r"""Pairwise `(N, N)` 3D Euclidean distance matrix in Ångströms."""
        return self.conformer.distance_matrix()

    def contact_csr(self, cutoff: float = 5.0, *, include_self: bool = False) -> CSR:
        r"""Sparse `(N, N)` [`CSR`][seroverse.core.graph.csr.CSR] spatial contact graph within `cutoff` Å."""
        return self.conformer.contact_csr(cutoff=cutoff, include_self=include_self)

    def rmsd_to(self, other: Structure | Conformer) -> float:
        r"""Optimal Kabsch-superimposed RMSD (in Å) to `other`."""
        target = other.conformer if isinstance(other, Structure) else other
        return self.conformer.rmsd_to(target)

    def align_to(self, reference: Structure | Conformer) -> Self:
        r"""Return a new `Structure` with its `conformer` optimally Kabsch-aligned onto `reference`."""
        target = reference.conformer if isinstance(reference, Structure) else reference
        return self.replace(conformer=self.conformer.align_to(target))


@dataclass(slots=True, frozen=True)
class Structures(Mols[Structure]):
    r"""Structure-of-Arrays (SoA) batch container for 3D [`Structure`][seroverse.core.mol.structure.Structure] records."""

    _scalar_type: ClassVar[type[Any] | None] = Structure

    node_types: Types.UInt32Array
    node_offsets: Types.UInt32Array
    edge_sources: Types.UInt32Array
    edge_targets: Types.UInt32Array
    edge_types: Types.UInt32Array
    edge_offsets: Types.UInt32Array
    canonicals: Types.BytesArray
    conformers: Conformers

    def centroids(self) -> Types.Float32Array2D:
        r"""Compute the `(len(self), 3)` geometric centroids across all structures in the batch."""
        return self.conformers.centroids()

    def radii_of_gyration(self) -> Types.Float32Array:
        r"""Compute the 1D `(len(self),)` radii of gyration ($R_g$ in Å) across all structures in the batch."""
        return self.conformers.radii_of_gyration()


# Kernels --------------------------------------------------------------------------------------------------------------
@Types.njit
def _conformer_centroid_kernel(
    x: Types.Float32Array,
    y: Types.Float32Array,
    z: Types.Float32Array,
) -> tuple[float, float, float]:
    r"""Compute the 3D centroid `(cx, cy, cz)` of a coordinate set."""
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    sx = 0.0
    sy = 0.0
    sz = 0.0
    for i in range(n):
        sx += float(x[i])
        sy += float(y[i])
        sz += float(z[i])
    inv_n = 1.0 / float(n)
    return sx * inv_n, sy * inv_n, sz * inv_n


@Types.njit
def _conformer_rg_kernel(
    x: Types.Float32Array,
    y: Types.Float32Array,
    z: Types.Float32Array,
) -> float:
    r"""Compute the unweighted 3D radius of gyration $R_g$ of a coordinate set."""
    n = len(x)
    if n <= 1:
        return 0.0
    cx, cy, cz = _conformer_centroid_kernel(x, y, z)
    sq_sum = 0.0
    for i in range(n):
        dx = float(x[i]) - cx
        dy = float(y[i]) - cy
        dz = float(z[i]) - cz
        sq_sum += dx * dx + dy * dy + dz * dz
    return float(np.sqrt(sq_sum / float(n)))


@Types.njit(parallel=True)
def _pairwise_dist3d_kernel(
    x: Types.Float32Array,
    y: Types.Float32Array,
    z: Types.Float32Array,
) -> Types.Float32Array2D:
    r"""Compute the symmetric `(N, N)` 3D Euclidean distance matrix in parallel."""
    n = len(x)
    out = np.zeros((n, n), dtype=np.float32)
    for i in prange(n):  # type: ignore[not-iterable]
        xi = float(x[i])
        yi = float(y[i])
        zi = float(z[i])
        for j in range(i + 1, n):
            dx = xi - float(x[j])
            dy = yi - float(y[j])
            dz = zi - float(z[j])
            d = np.float32(np.sqrt(dx * dx + dy * dy + dz * dz))
            out[i, j] = d
            out[j, i] = d
    return out


@Types.njit
def _contact_edges_kernel(
    x: Types.Float32Array,
    y: Types.Float32Array,
    z: Types.Float32Array,
    cutoff: float,
    include_self: bool,
) -> tuple[Types.UInt32Array, Types.UInt32Array, Types.Float32Array]:
    r"""Two-pass spatial contact edge builder within `cutoff` Å."""
    n = len(x)
    cutoff_sq = cutoff * cutoff
    count = 0
    for i in range(n):
        xi = float(x[i])
        yi = float(y[i])
        zi = float(z[i])
        if include_self:
            count += 1
        for j in range(i + 1, n):
            dx = xi - float(x[j])
            dy = yi - float(y[j])
            dz = zi - float(z[j])
            if dx * dx + dy * dy + dz * dz <= cutoff_sq:
                count += 2

    rows = np.empty(count, dtype=np.uint32)
    cols = np.empty(count, dtype=np.uint32)
    dists = np.empty(count, dtype=np.float32)
    if count == 0:
        return rows, cols, dists

    cur = 0
    for i in range(n):
        xi = float(x[i])
        yi = float(y[i])
        zi = float(z[i])
        if include_self:
            rows[cur] = np.uint32(i)
            cols[cur] = np.uint32(i)
            dists[cur] = np.float32(0.0)
            cur += 1
        for j in range(i + 1, n):
            dx = xi - float(x[j])
            dy = yi - float(y[j])
            dz = zi - float(z[j])
            d2 = dx * dx + dy * dy + dz * dz
            if d2 <= cutoff_sq:
                d = np.float32(np.sqrt(d2))
                rows[cur] = np.uint32(i)
                cols[cur] = np.uint32(j)
                dists[cur] = d
                cur += 1
                rows[cur] = np.uint32(j)
                cols[cur] = np.uint32(i)
                dists[cur] = d
                cur += 1
    return rows, cols, dists


@Types.njit
def _infer_covalent_bonds_kernel(
    x: Types.Float32Array,
    y: Types.Float32Array,
    z: Types.Float32Array,
    elem_ids: Types.UInt32Array,
    covalent_radii: Types.Float32Array,
    tolerance: float,
) -> tuple[Types.UInt32Array, Types.UInt32Array]:
    r"""Two-pass covalent bond inference from 3D coordinates and element covalent radii."""
    n = len(x)
    min_dist_sq = 0.16  # 0.4 Å minimum interatomic distance
    count = 0
    for i in range(n):
        xi = float(x[i])
        yi = float(y[i])
        zi = float(z[i])
        ri = float(covalent_radii[int(elem_ids[i])])
        for j in range(i + 1, n):
            max_d = ri + float(covalent_radii[int(elem_ids[j])]) + tolerance
            dx = xi - float(x[j])
            dy = yi - float(y[j])
            dz = zi - float(z[j])
            d2 = dx * dx + dy * dy + dz * dz
            if min_dist_sq <= d2 <= max_d * max_d:
                count += 1

    sources = np.empty(count, dtype=np.uint32)
    targets = np.empty(count, dtype=np.uint32)
    if count == 0:
        return sources, targets

    cur = 0
    for i in range(n):
        xi = float(x[i])
        yi = float(y[i])
        zi = float(z[i])
        ri = float(covalent_radii[int(elem_ids[i])])
        for j in range(i + 1, n):
            max_d = ri + float(covalent_radii[int(elem_ids[j])]) + tolerance
            dx = xi - float(x[j])
            dy = yi - float(y[j])
            dz = zi - float(z[j])
            d2 = dx * dx + dy * dy + dz * dz
            if min_dist_sq <= d2 <= max_d * max_d:
                sources[cur] = np.uint32(i)
                targets[cur] = np.uint32(j)
                cur += 1
    return sources, targets


@Types.njit
def _kabsch_align_kernel(
    px: Types.Float32Array,
    py: Types.Float32Array,
    pz: Types.Float32Array,
    qx: Types.Float32Array,
    qy: Types.Float32Array,
    qz: Types.Float32Array,
) -> tuple[float, Types.Float32Array, Types.Float32Array, Types.Float32Array]:
    r"""Optimal 3D Kabsch rotation and translation of `(px, py, pz)` onto `(qx, qy, qz)`."""
    n = len(px)
    ax = np.empty(n, dtype=np.float32)
    ay = np.empty(n, dtype=np.float32)
    az = np.empty(n, dtype=np.float32)
    if n == 0:
        return 0.0, ax, ay, az

    pcx, pcy, pcz = _conformer_centroid_kernel(px, py, pz)
    qcx, qcy, qcz = _conformer_centroid_kernel(qx, qy, qz)

    p_mat = np.empty((n, 3), dtype=np.float64)
    q_mat = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        p_mat[i, 0] = float(px[i]) - pcx
        p_mat[i, 1] = float(py[i]) - pcy
        p_mat[i, 2] = float(pz[i]) - pcz
        q_mat[i, 0] = float(qx[i]) - qcx
        q_mat[i, 1] = float(qy[i]) - qcy
        q_mat[i, 2] = float(qz[i]) - qcz

    # Covariance matrix H = P^T @ Q (3x3)
    h00 = 0.0
    h01 = 0.0
    h02 = 0.0
    h10 = 0.0
    h11 = 0.0
    h12 = 0.0
    h20 = 0.0
    h21 = 0.0
    h22 = 0.0
    for i in range(n):
        p0, p1, p2 = p_mat[i, 0], p_mat[i, 1], p_mat[i, 2]
        q0, q1, q2 = q_mat[i, 0], q_mat[i, 1], q_mat[i, 2]
        h00 += p0 * q0
        h01 += p0 * q1
        h02 += p0 * q2
        h10 += p1 * q0
        h11 += p1 * q1
        h12 += p1 * q2
        h20 += p2 * q0
        h21 += p2 * q1
        h22 += p2 * q2

    # 4x4 symmetric Horn quaternion matrix F such that q^T F q = sum_{r,c} H_{r,c} R(q)_{r,c}
    f = np.empty((4, 4), dtype=np.float64)
    f[0, 0] = h00 + h11 + h22
    f[0, 1] = f[1, 0] = h21 - h12
    f[0, 2] = f[2, 0] = h02 - h20
    f[0, 3] = f[3, 0] = h10 - h01
    f[1, 1] = h00 - h11 - h22
    f[1, 2] = f[2, 1] = h01 + h10
    f[1, 3] = f[3, 1] = h02 + h20
    f[2, 2] = -h00 + h11 - h22
    f[2, 3] = f[3, 2] = h12 + h21
    f[3, 3] = -h00 - h11 + h22

    # Solve 4x4 symmetric eigenproblem via cyclic Jacobi rotations (zero SciPy / LAPACK)
    v_eig = np.eye(4, dtype=np.float64)
    for _ in range(25):
        max_off = 0.0
        for p in range(3):
            for q in range(p + 1, 4):
                apq = abs(f[p, q])
                if apq > max_off:
                    max_off = apq
        if max_off < 1e-14:
            break
        for p in range(3):
            for q in range(p + 1, 4):
                apq = f[p, q]
                if abs(apq) < 1e-15:
                    continue
                diff = f[q, q] - f[p, p]
                if abs(apq) <= 1e-15 * abs(diff):
                    t = apq / diff
                else:
                    tau = diff / (2.0 * apq)
                    t = (1.0 if tau >= 0.0 else -1.0) / (abs(tau) + np.sqrt(1.0 + tau * tau))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = t * c
                tau_rot = s / (1.0 + c)
                f[p, p] -= t * apq
                f[q, q] += t * apq
                f[p, q] = 0.0
                f[q, p] = 0.0
                for r in range(4):
                    if r != p and r != q:
                        frp = f[r, p]
                        frq = f[r, q]
                        f[r, p] = f[p, r] = frp - s * (frq + tau_rot * frp)
                        f[r, q] = f[q, r] = frq + s * (frp - tau_rot * frq)
                for r in range(4):
                    vrp = v_eig[r, p]
                    vrq = v_eig[r, q]
                    v_eig[r, p] = vrp - s * (vrq + tau_rot * vrp)
                    v_eig[r, q] = vrq + s * (vrp - tau_rot * vrq)

    # Select unit quaternion eigenvector corresponding to maximum eigenvalue
    best_col = 0
    for col in range(1, 4):
        if f[col, col] > f[best_col, best_col]:
            best_col = col
    qw = v_eig[0, best_col]
    qx_q = v_eig[1, best_col]
    qy_q = v_eig[2, best_col]
    qz_q = v_eig[3, best_col]

    rot = np.empty((3, 3), dtype=np.float64)
    rot[0, 0] = qw * qw + qx_q * qx_q - qy_q * qy_q - qz_q * qz_q
    rot[0, 1] = 2.0 * (qx_q * qy_q - qw * qz_q)
    rot[0, 2] = 2.0 * (qx_q * qz_q + qw * qy_q)
    rot[1, 0] = 2.0 * (qx_q * qy_q + qw * qz_q)
    rot[1, 1] = qw * qw - qx_q * qx_q + qy_q * qy_q - qz_q * qz_q
    rot[1, 2] = 2.0 * (qy_q * qz_q - qw * qx_q)
    rot[2, 0] = 2.0 * (qx_q * qz_q - qw * qy_q)
    rot[2, 1] = 2.0 * (qy_q * qz_q + qw * qx_q)
    rot[2, 2] = qw * qw - qx_q * qx_q - qy_q * qy_q + qz_q * qz_q

    sq_err = 0.0
    for i in range(n):
        p0 = p_mat[i, 0]
        p1 = p_mat[i, 1]
        p2 = p_mat[i, 2]
        rx = p0 * rot[0, 0] + p1 * rot[1, 0] + p2 * rot[2, 0]
        ry = p0 * rot[0, 1] + p1 * rot[1, 1] + p2 * rot[2, 1]
        rz = p0 * rot[0, 2] + p1 * rot[1, 2] + p2 * rot[2, 2]
        dx = rx - q_mat[i, 0]
        dy = ry - q_mat[i, 1]
        dz = rz - q_mat[i, 2]
        sq_err += dx * dx + dy * dy + dz * dz
        ax[i] = np.float32(rx + qcx)
        ay[i] = np.float32(ry + qcy)
        az[i] = np.float32(rz + qcz)

    rmsd = float(np.sqrt(sq_err / float(n)))
    return rmsd, ax, ay, az


@Types.njit(parallel=True)
def _batch_centroids_kernel(
    x: Types.Float32Array,
    y: Types.Float32Array,
    z: Types.Float32Array,
    offsets: Types.Int32Array,
    lengths: Types.Int32Array,
) -> Types.Float32Array2D:
    r"""Compute 3D centroids across a `Conformers` ragged batch in parallel."""
    b = len(offsets)
    out = np.zeros((b, 3), dtype=np.float32)
    for i in prange(b):  # type: ignore[not-iterable]
        cnt = int(lengths[i])
        if cnt == 0:
            continue
        off = int(offsets[i])
        cx, cy, cz = _conformer_centroid_kernel(x[off : off + cnt], y[off : off + cnt], z[off : off + cnt])
        out[i, 0] = np.float32(cx)
        out[i, 1] = np.float32(cy)
        out[i, 2] = np.float32(cz)
    return out


@Types.njit(parallel=True)
def _batch_rg_kernel(
    x: Types.Float32Array,
    y: Types.Float32Array,
    z: Types.Float32Array,
    offsets: Types.Int32Array,
    lengths: Types.Int32Array,
) -> Types.Float32Array:
    r"""Compute radii of gyration ($R_g$) across a `Conformers` ragged batch in parallel."""
    b = len(offsets)
    out = np.zeros(b, dtype=np.float32)
    for i in prange(b):  # type: ignore[not-iterable]
        cnt = int(lengths[i])
        if cnt <= 1:
            continue
        off = int(offsets[i])
        out[i] = np.float32(_conformer_rg_kernel(x[off : off + cnt], y[off : off + cnt], z[off : off + cnt]))
    return out
