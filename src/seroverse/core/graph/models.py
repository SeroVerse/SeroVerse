"""Core graph representations, bitfield packing, and vectorized SoA containers."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Self, cast, overload

import numpy as np
from numba import njit, prange

from seroverse.core.collections import BatchedContainer, ScalarView
from seroverse.core.graph.csr import CSR, RelationalCSR

# Constants ------------------------------------------------------------------------------------------------------------
TAG_EDGE: int = 1 << 31  # 0x80000000 (Bit 31 discriminator for injected edge nodes)


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class BitField:
    """
    Descriptor defining a bitfield slice (shift, width, mask).
    Provides safe packing, extraction, and bitmask computation.
    """

    shift: int
    width: int

    @property
    def mask(self) -> int:
        """Absolute bitmask positioned at field shift."""
        return ((1 << self.width) - 1) << self.shift

    @property
    def max_val(self) -> int:
        """Maximum unsigned value that fits within this bitfield width."""
        return (1 << self.width) - 1

    def extract(self, val: int) -> int:
        """Extracts and right-shifts this bitfield's value from an integer."""
        return (val >> self.shift) & self.max_val

    def extract_array(self, arr: np.ndarray) -> np.ndarray:
        """Extracts and right-shifts this bitfield's value across a numpy array."""
        return (arr >> self.shift) & self.max_val

    def pack(self, field_val: int) -> int:
        """Validates and shifts field_val into its bit position."""
        if not (0 <= field_val <= self.max_val):
            raise ValueError(f"Value {field_val} exceeds {self.width} bits (max {self.max_val})")
        return (field_val & self.max_val) << self.shift

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        """Descriptor access: returns self on class access, extracted field integer on instance access."""
        if instance is None:
            return self
        return (int(instance) >> self.shift) & self.max_val


class NodeBase(ABC):
    """Abstract base protocol for all core nodes."""

    @property
    @abstractmethod
    def type_id(self) -> int:
        """Semantic integer label used in core adjacency matrices and embeddings."""
        ...

    @property
    def is_edge_node(self) -> bool:
        """Returns True if this node represents an injected edge in a bipartite core."""
        return bool(self.type_id & TAG_EDGE)


class IntNode(int, NodeBase):
    """Base model for core nodes represented as bit-packed primitive integers.

    Inherits from Python's built-in `int` and `GraphNode`, enabling zero-allocation
    integer node semantics across NumPy arrays, SciPy CSR matrices, and core algorithms.
    Hosts BitField member descriptors that automatically extract bit values when accessed
    on instances, while providing bitmasking, range validation, and packing at the class level.
    """

    _fields: dict[str, BitField]
    _total_bits: int = 32

    def __init_subclass__(cls, total_bits: int = 32, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._total_bits = total_bits
        fields: dict[str, BitField] = {}
        occupied_mask = 0

        for attr_name, attr_val in cls.__dict__.items():
            if isinstance(attr_val, BitField):
                name_upper = attr_name.upper()
                if attr_val.shift + attr_val.width > total_bits:
                    raise ValueError(
                        f"{cls.__name__}.{attr_name} (shift={attr_val.shift}, width={attr_val.width}) "
                        f"exceeds total {total_bits} bits."
                    )
                field_mask = attr_val.mask
                if (occupied_mask & field_mask) and not any(
                    f.shift == attr_val.shift and f.width == attr_val.width for f in fields.values()
                ):
                    raise ValueError(
                        f"BitField overlap detected in {cls.__name__} for field '{attr_name}' "
                        f"(shift={attr_val.shift}, width={attr_val.width})."
                    )
                occupied_mask |= field_mask
                fields[name_upper] = attr_val

        cls._fields = fields

    @property
    def type_id(self) -> int:
        return int(self)

    @property
    def is_edge_node(self) -> bool:
        return bool(int(self) & TAG_EDGE)

    @classmethod
    def get_field(cls, field_name: str) -> BitField:
        name = field_name.upper()
        if name not in cls._fields:
            raise AttributeError(f"{cls.__name__} has no BitField named '{field_name}'.")
        return cls._fields[name]

    @classmethod
    def mask(cls, *field_names: str) -> int:
        """Constructs an absolute bitmask covering the specified bitfield names."""
        m = 0
        for name in field_names:
            m |= cls.get_field(name).mask
        return m

    @classmethod
    def mask_excluding(cls, *field_names: str) -> int:
        """Constructs a bitmask excluding the specified bitfield names."""
        full_mask = (1 << cls._total_bits) - 1
        return full_mask & ~cls.mask(*field_names)

    @classmethod
    def pack(cls, **kwargs: int) -> int:
        """Packs keyword field values into a composite integer."""
        val = 0
        for name, field_val in kwargs.items():
            val |= cls.get_field(name).pack(field_val)
        return val

    @classmethod
    def extract(cls, val: int, field_name: str) -> int:
        return cls.get_field(field_name).extract(val)

    @classmethod
    def extract_array(cls, arr: np.ndarray, field_name: str) -> np.ndarray:
        return cls.get_field(field_name).extract_array(arr)


@dataclass(slots=True, frozen=True)
class DirectedEdge:
    """Base class for all directed core edges."""

    source: int
    target: int
    edge_type: int = 1

    def __post_init__(self) -> None:
        if self.source < 0 or self.target < 0:
            raise ValueError(
                f"DirectedEdge source and target must be non-negative node indices, got source={self.source}, target={self.target}"
            )


@dataclass(slots=True, frozen=True)
class Graph(ScalarView):
    """
    Scalar view and topological model representing a single directed graph.
    Provides sparse CSR conversion, multi-hop [`RelationalCSR`][seroverse.core.graph.csr.RelationalCSR]
    construction, Laplacian spectrum calculation, and 1-Wasserstein spectral distance.
    """

    nodes: tuple[Any, ...] = ()
    edges: tuple[DirectedEdge, ...] = ()

    @cache  # noqa: B019
    def as_graph(
        self,
        as_bipartite: bool = False,
        symmetrize: bool = False,
    ) -> tuple[CSR, np.ndarray[tuple[int], np.dtype[np.uint32]]]:
        """
        Builds a sparse CSR adjacency matrix and node label array.

        All biological and genomic models in KBank are strictly directed graphs.
        By default, returns the canonical directed representation. If a mathematical
        algorithm requires a symmetric input matrix (e.g. Laplacian spectrum calculation
        or bidirectional message passing), specify `symmetrize=True`.

        Args:
            as_bipartite: If True, treats each directed edge as an intermediate node,
                forming a bipartite incidence core where edge weights are 1.
                If False, edges directly connect primary nodes with edge weights equal
                to the integer edge type.
            symmetrize: If True, produces a symmetrized (bidirectional) adjacency matrix projection.

        Returns:
            A tuple of:
                - adj: Sparse CSR adjacency matrix of shape (total_nodes, total_nodes).
                - labels: 1D NumPy array of node labels (`np.ndarray[tuple[int], np.dtype[np.uint32]]`).
        """
        is_directed = not symmetrize

        nodes = self.nodes
        edges = self.edges
        num_primary = len(nodes)
        num_edges = len(edges)
        total_nodes = num_primary + num_edges if as_bipartite else num_primary

        if total_nodes == 0:
            return CSR.empty((0, 0), dtype=np.uint32), np.empty(0, dtype=np.uint32)

        labels = np.empty(total_nodes, dtype=np.uint32)
        for i, node in enumerate(nodes):
            labels[i] = node.type_id if hasattr(node, "type_id") else int(node)

        if num_edges == 0:
            return CSR.empty((total_nodes, total_nodes), dtype=np.uint32), labels

        sources = np.fromiter((e.source for e in edges), dtype=np.uint32, count=num_edges)
        targets = np.fromiter((e.target for e in edges), dtype=np.uint32, count=num_edges)
        edge_types = np.fromiter((int(e.edge_type) for e in edges), dtype=np.uint32, count=num_edges)

        if as_bipartite:
            edge_node_ids = np.arange(num_primary, total_nodes, dtype=np.uint32)
            labels[num_primary:total_nodes] = edge_types

            if is_directed:
                rows = np.concatenate([sources, edge_node_ids])
                cols = np.concatenate([edge_node_ids, targets])
                data = np.ones(2 * num_edges, dtype=np.uint32)
            else:
                rows = np.concatenate([sources, edge_node_ids, targets, edge_node_ids])
                cols = np.concatenate([edge_node_ids, sources, edge_node_ids, targets])
                data = np.ones(4 * num_edges, dtype=np.uint32)
        else:
            if is_directed:
                rows = sources
                cols = targets
                data = edge_types
            else:
                rows = np.concatenate([sources, targets])
                cols = np.concatenate([targets, sources])
                data = np.concatenate([edge_types, edge_types])

        adj = CSR.from_coo(rows, cols, data, shape=(total_nodes, total_nodes), dtype=np.uint32)
        return adj, labels

    def as_relational_csr(
        self,
        *,
        include_2hop: bool = True,
        add_virtual_pool: bool = True,
    ) -> RelationalCSR:
        """Build a multi-hop [`RelationalCSR`][seroverse.core.graph.csr.RelationalCSR] for this graph."""
        return RelationalCSR.from_graph(
            self,
            include_2hop=include_2hop,
            add_virtual_pool=add_virtual_pool,
        )

    def __len__(self) -> int:
        return len(self.nodes)

    @cache  # noqa: B019
    def laplacian_spectrum(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Computes the continuous topological spectrum of the core.

        Constructs the unweighted, symmetrized normalized core Laplacian via
        [`CSR.laplacian`][seroverse.core.graph.csr.CSR.laplacian] and solves for the sorted
        eigenvalues using :func:`numpy.linalg.eigvalsh`.

        Returns:
            A 1D sorted NumPy array of eigenvalues (`np.ndarray[tuple[int], np.dtype[np.float64]]`).
        """
        adj, _ = self.as_graph(as_bipartite=False)
        if adj.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        L_dense = adj.laplacian(normed=True, symmetrize_unweighted=True)
        return np.sort(np.linalg.eigvalsh(L_dense))

    def spectral_distance(self, other: Graph) -> float:
        """
        Computes the topological Wasserstein Distance (Earth Mover's Distance)
        between this core model and another, comparing skeletal shapes.

        Extracts the Laplacian spectra from both graphs and computes their
        Wasserstein-1 distance via a compiled Numba 1D optimal transport kernel.

        Args:
            other: Another GraphModel instance to compare against.

        Returns:
            Wasserstein distance between the two normalized Laplacian spectra as a float.
        """
        spec1 = self.laplacian_spectrum()
        spec2 = other.laplacian_spectrum()
        return float(_wasserstein_1d_sorted_kernel(spec1, spec2))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self.nodes)} nodes, {len(self.edges)} edges)"


@dataclass(slots=True, frozen=True)
class Graphs[T: Graph = Graph, S: Graphs[Any, Any] = Graphs[Any, Any]](BatchedContainer[T, S]):
    """
    Base Structure-of-Arrays (SoA) container for batched graph datasets.
    Provides block-diagonal CSR adjacency, Wasserstein spectral distance matrices,
    pan-graph compression, and vectorized SoA array slicing and concatenation.
    """

    node_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    node_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_sources: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_targets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]]

    def __len__(self) -> int:
        return len(self.node_offsets) - 1

    @property
    def total_nodes(self) -> int:
        return len(self.node_types)

    @property
    def total_edges(self) -> int:
        return len(self.edge_sources)

    @property
    def node_lengths(self) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
        """Number of nodes in each graph in the batch."""
        return np.diff(self.node_offsets)

    @property
    def edge_lengths(self) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
        """Number of edges in each graph in the batch."""
        return np.diff(self.edge_offsets)

    @classmethod
    def empty(cls) -> Self:
        """Create an empty 0-length graph batch."""
        return cls(
            node_types=np.empty(0, dtype=np.uint32),
            node_offsets=np.zeros(1, dtype=np.uint32),
            edge_sources=np.empty(0, dtype=np.uint32),
            edge_targets=np.empty(0, dtype=np.uint32),
            edge_types=np.empty(0, dtype=np.uint32),
            edge_offsets=np.zeros(1, dtype=np.uint32),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenate multiple graph batches into a single contiguous batch."""
        batch_list: list[Self] = [b for b in batches if len(b) > 0]
        if not batch_list:
            return cls.empty()
        if len(batch_list) == 1:
            return batch_list[0]

        nodes, n_offs, sources, targets, types, e_offs = cls._concat_soa_base(batch_list)
        return cls(
            node_types=nodes,
            node_offsets=n_offs,
            edge_sources=sources,
            edge_targets=targets,
            edge_types=types,
            edge_offsets=e_offs,
        )

    @classmethod
    def from_list(cls, graphs: Sequence[Graph]) -> Self:
        """Pack a sequence of [`GraphModel`][seroverse.core.graph.models.GraphModel] instances into a `GraphBatch`."""
        n_graphs = len(graphs)
        if n_graphs == 0:
            return cls.empty()

        node_offsets = np.zeros(n_graphs + 1, dtype=np.uint32)
        edge_offsets = np.zeros(n_graphs + 1, dtype=np.uint32)
        tot_nodes = sum(len(g.nodes) for g in graphs)
        tot_edges = sum(len(g.edges) for g in graphs)

        node_types = np.empty(tot_nodes, dtype=np.uint32)
        edge_sources = np.empty(tot_edges, dtype=np.uint32)
        edge_targets = np.empty(tot_edges, dtype=np.uint32)
        edge_types = np.empty(tot_edges, dtype=np.uint32)

        curr_n = 0
        curr_e = 0
        for i, g in enumerate(graphs):
            gn = len(g.nodes)
            ge = len(g.edges)
            for j, node in enumerate(g.nodes):
                node_types[curr_n + j] = node.type_id if hasattr(node, "type_id") else int(node)
            for j, edge in enumerate(g.edges):
                edge_sources[curr_e + j] = edge.source + curr_n
                edge_targets[curr_e + j] = edge.target + curr_n
                edge_types[curr_e + j] = int(edge.edge_type)
            curr_n += gn
            curr_e += ge
            node_offsets[i + 1] = curr_n
            edge_offsets[i + 1] = curr_e

        return cls(
            node_types=node_types,
            node_offsets=node_offsets,
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            edge_types=edge_types,
            edge_offsets=edge_offsets,
        )

    @classmethod
    def random(
        cls,
        n_graphs: int = 10,
        n_nodes: int | tuple[int, int] = (5, 15),
        p: float = 0.25,
        *,
        n_node_types: int = 8,
        n_edge_types: int = 4,
        directed: bool = True,
        allow_self_loops: bool = False,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        r"""Generate a random [`GraphBatch`][seroverse.core.graph.models.GraphBatch] using the Erdős–Rényi $G(n, p)$ model.

        Args:
            n_graphs: Number of graphs in the batch (`n_graphs >= 0`). Defaults to `10`.
            n_nodes: Either a fixed integer node count per graph or an inclusive `(min_nodes, max_nodes)`
                2-tuple sampled uniformly per graph. Defaults to `(5, 15)`.
            p: Edge inclusion probability $p \in [0, 1]$. Defaults to `0.25`.
            n_node_types: Number of distinct integer node types sampled uniformly in `[1, n_node_types]`. Defaults to `8`.
            n_edge_types: Number of distinct integer edge types sampled uniformly in `[1, n_edge_types]`. Defaults to `4`.
            directed: If `True` (default), generates independent directed edges $(u \to v)$. If `False`,
                generates symmetric undirected edges $(u \leftrightarrow v)$.
            allow_self_loops: If `True`, permits self-loops $(u \to u)$. Defaults to `False`.
            rng: Optional `np.random.Generator`, integer seed, or `None` passed to `np.random.default_rng`.

        Returns:
            Self: A newly generated `GraphBatch` instance.
        """
        if n_graphs < 0:
            raise ValueError(f"n_graphs must be >= 0; got {n_graphs}.")
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Edge probability p must be in [0.0, 1.0]; got {p}.")
        if n_node_types < 1 or n_edge_types < 1:
            raise ValueError(f"n_node_types ({n_node_types}) and n_edge_types ({n_edge_types}) must be >= 1.")
        if n_graphs == 0:
            return cls.empty()

        gen = np.random.default_rng(rng)
        if isinstance(n_nodes, (int, np.integer)):
            if int(n_nodes) < 0:
                raise ValueError(f"n_nodes must be >= 0; got {n_nodes}.")
            node_lens = np.full(n_graphs, int(n_nodes), dtype=np.uint32)
        else:
            min_n, max_n = int(n_nodes[0]), int(n_nodes[1])
            if min_n < 0 or max_n < min_n:
                raise ValueError(f"Invalid n_nodes range ({min_n}, {max_n}); require 0 <= min_nodes <= max_nodes.")
            node_lens = gen.integers(min_n, max_n + 1, size=n_graphs, dtype=np.uint32)

        node_offsets = np.zeros(n_graphs + 1, dtype=np.uint32)
        np.cumsum(node_lens, dtype=np.uint32, out=node_offsets[1:])
        total_nodes = int(node_offsets[-1])
        node_types = gen.integers(1, n_node_types + 1, size=total_nodes, dtype=np.uint32)

        seed = np.uint64(gen.integers(1, (1 << 63) - 1, dtype=np.uint64))
        edge_sources, edge_targets, edge_types, edge_offsets = _erdos_renyi_edges_kernel(
            node_offsets,
            float(p),
            int(n_edge_types),
            bool(directed),
            bool(allow_self_loops),
            seed,
        )
        return cls(
            node_types=node_types,
            node_offsets=node_offsets,
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            edge_types=edge_types,
            edge_offsets=edge_offsets,
        )

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> T: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> S: ...

    @overload
    def __getitem__(self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]) -> T | S: ...

    def __getitem__(self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]) -> T | S:
        """Extract a single [`GraphModel`][seroverse.core.graph.models.GraphModel] by integer index, or a sub-batch by slice/mask/indices."""
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            n_graphs = len(self)
            if idx < 0:
                idx += n_graphs
            if idx < 0 or idx >= n_graphs:
                raise IndexError(f"GraphBatch index {item} out of bounds for batch of length {n_graphs}.")
            n_s, n_e = int(self.node_offsets[idx]), int(self.node_offsets[idx + 1])
            e_s, e_e = int(self.edge_offsets[idx]), int(self.edge_offsets[idx + 1])
            nodes = tuple(int(x) for x in self.node_types[n_s:n_e])
            edges = tuple(
                DirectedEdge(
                    source=int(self.edge_sources[k]) - n_s,
                    target=int(self.edge_targets[k]) - n_s,
                    edge_type=int(self.edge_types[k]),
                )
                for k in range(e_s, e_e)
            )
            return cast(T, Graph(nodes=nodes, edges=edges))

        indices = self._normalize_indices(item)
        if len(indices) == 0:
            return cast(S, self.empty())
        nodes_arr, n_offs, sources, targets, types, e_offs = self._slice_soa_arrays(indices)
        return cast(
            S,
            self.__class__(
                node_types=nodes_arr,
                node_offsets=n_offs,
                edge_sources=sources,
                edge_targets=targets,
                edge_types=types,
                edge_offsets=e_offs,
            ),
        )

    def get_block_diagonal_graph(self, unweighted: bool = False) -> CSR:
        """
        Returns a single disconnected CSR core representing the entire batch.

        Args:
            unweighted: If True, sets all non-zero edge weights to 1;
                otherwise retains the integer edge types.

        Returns:
            Block-diagonal sparse CSR adjacency matrix of shape (total_nodes, total_nodes).
        """
        N = len(self.node_types)
        if N == 0 or len(self.edge_sources) == 0:
            return CSR.empty((N, N), dtype=np.uint32)

        weights = np.ones(len(self.edge_types), dtype=np.uint32) if unweighted else self.edge_types
        return CSR.from_coo(self.edge_sources, self.edge_targets, weights, shape=(N, N), dtype=np.uint32)

    def as_relational_csr(
        self,
        edge_orientations: Sequence[int] | np.ndarray | None = None,
        *,
        include_2hop: bool = True,
        add_virtual_pool: bool = True,
    ) -> RelationalCSR:
        """Build a multi-hop [`RelationalCSR`][seroverse.core.graph.csr.RelationalCSR] for this batch."""
        return RelationalCSR.from_batch(
            self,
            edge_orientations=edge_orientations,
            include_2hop=include_2hop,
            add_virtual_pool=add_virtual_pool,
        )

    def spectral_distance_matrix(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        Computes the all-vs-all continuous topological distance matrix across the batch.

        Extracts per-core normalized Laplacian spectra and computes pairwise
        Wasserstein-1 distances in parallel using compiled Numba kernels.

        Returns:
            A dense symmetric matrix (N x N) of Wasserstein distances (`np.ndarray[tuple[int, int], np.dtype[np.float64]]`).
        """
        return _batch_spectral_distance_matrix_kernel(
            self.node_offsets,
            self.edge_sources,
            self.edge_targets,
            self.edge_offsets,
        )

    def as_pan_graph(
        self,
        node_mask: int | None = None,
    ) -> tuple[
        CSR,
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
    ]:
        """
        Compresses the SoA batch into a unified directed pan-core network.
        Merges identical nodes, records node occurrence frequencies,
        and computes the exact occurrence frequency (count) of each directed transition.

        Pan-graphs in KBank are strictly directed consensus networks representing
        synteny flow and biosynthetic transitions.

        Args:
            node_mask: Optional bitmask to apply to node types before merging.

        Returns:
            A tuple of:
                - adj: Sparse CSR pan-core adjacency matrix of shape (num_pan_nodes, num_pan_nodes)
                  where data values represent transition occurrence counts.
                - pan_nodes: 1D array of pan-node integer labels (`np.ndarray[tuple[int], np.dtype[np.uint32]]`).
                - node_counts: 1D array of frequencies indicating how many original nodes merged into
                  each pan-node (`np.ndarray[tuple[int], np.dtype[np.uint32]]`).
        """
        N = len(self.node_types)
        if N == 0:
            return CSR.empty((0, 0), dtype=np.uint32), np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32)

        masked_types = (self.node_types & node_mask) if node_mask is not None else self.node_types
        pan_nodes, inverse_indices, node_counts = np.unique(masked_types, return_inverse=True, return_counts=True)
        num_pan_nodes = len(pan_nodes)

        if len(self.edge_sources) == 0:
            return CSR.empty((num_pan_nodes, num_pan_nodes), dtype=np.uint32), pan_nodes, node_counts.astype(np.uint32)

        p_src = inverse_indices[self.edge_sources]
        p_tgt = inverse_indices[self.edge_targets]

        adj = CSR.from_coo(
            p_src,
            p_tgt,
            np.ones(len(p_src), dtype=np.uint32),
            shape=(num_pan_nodes, num_pan_nodes),
            dtype=np.uint32,
            sum_duplicates=True,
        )
        return adj, pan_nodes, node_counts.astype(np.uint32)

    def _slice_soa_arrays(
        self, kept_indices: np.ndarray[tuple[int], np.dtype[np.integer[Any]]]
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
    ]:
        """
        Vectorized helper to slice the core SoA arrays for a subset of graphs.
        """
        num_kept = len(kept_indices)
        if num_kept == 0:
            return (
                np.array([], dtype=np.uint32),
                np.zeros(1, dtype=np.uint32),
                np.array([], dtype=np.uint32),
                np.array([], dtype=np.uint32),
                np.array([], dtype=np.uint32),
                np.zeros(1, dtype=np.uint32),
            )

        idx = np.ascontiguousarray(kept_indices, dtype=np.int64)
        node_lens = self.node_offsets[idx + 1] - self.node_offsets[idx]
        edge_lens = self.edge_offsets[idx + 1] - self.edge_offsets[idx]

        new_node_offsets = np.empty(num_kept + 1, dtype=np.uint32)
        new_node_offsets[0] = 0
        np.cumsum(node_lens, dtype=np.uint32, out=new_node_offsets[1:])

        new_edge_offsets = np.empty(num_kept + 1, dtype=np.uint32)
        new_edge_offsets[0] = 0
        np.cumsum(edge_lens, dtype=np.uint32, out=new_edge_offsets[1:])

        new_nodes, new_sources, new_targets, new_types = _slice_graph_batch_kernel(
            idx,
            self.node_types,
            self.node_offsets,
            self.edge_sources,
            self.edge_targets,
            self.edge_types,
            self.edge_offsets,
            new_node_offsets,
            new_edge_offsets,
        )
        return (
            new_nodes,
            new_node_offsets,
            new_sources,
            new_targets,
            new_types,
            new_edge_offsets,
        )

    @classmethod
    def _concat_soa_base(
        cls, batches: Sequence[Any]
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
    ]:
        """
        Fast vectorized concatenation of core SoA arrays across multiple batches.
        Shifts node_offsets, edge_offsets, and rebases edge_sources and edge_targets.
        """
        if not batches:
            return (
                np.empty(0, dtype=np.uint32),
                np.zeros(1, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.zeros(1, dtype=np.uint32),
            )

        new_node_types = np.concatenate([b.node_types for b in batches])
        new_edge_types = np.concatenate([b.edge_types for b in batches])

        total_graphs = sum(len(b) for b in batches)
        new_node_offsets = np.empty(total_graphs + 1, dtype=np.uint32)
        new_node_offsets[0] = 0
        new_edge_offsets = np.empty(total_graphs + 1, dtype=np.uint32)
        new_edge_offsets[0] = 0

        sources_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []
        node_shift = 0
        edge_shift = 0
        curr_g = 0

        for b in batches:
            n_g = len(b)
            if n_g > 0:
                new_node_offsets[curr_g + 1 : curr_g + 1 + n_g] = b.node_offsets[1:] + node_shift
                new_edge_offsets[curr_g + 1 : curr_g + 1 + n_g] = b.edge_offsets[1:] + edge_shift
                curr_g += n_g

            if len(b.edge_sources) > 0:
                sources_list.append(b.edge_sources + node_shift)
                targets_list.append(b.edge_targets + node_shift)

            node_shift += len(b.node_types)
            edge_shift += len(b.edge_types)

        new_sources = np.concatenate(sources_list) if sources_list else np.empty(0, dtype=np.uint32)
        new_targets = np.concatenate(targets_list) if targets_list else np.empty(0, dtype=np.uint32)

        return (
            new_node_types,
            new_node_offsets,
            new_sources,
            new_targets,
            new_edge_types,
            new_edge_offsets,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self):,} items | {self.total_nodes:,} nodes, {self.total_edges:,} edges)"
        )


# Kernels --------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _wasserstein_1d_sorted_kernel(u: np.ndarray, v: np.ndarray) -> float:
    """Compute exact 1-Wasserstein distance between two pre-sorted 1D empirical distributions."""
    n = len(u)
    m = len(v)
    if n == 0 or m == 0:
        return 0.0 if n == m else 1.0

    inv_n = 1.0 / float(n)
    inv_m = 1.0 / float(m)
    i = 0
    j = 0
    dist = 0.0
    curr_x = u[0] if u[0] < v[0] else v[0]

    while i < n or j < m:
        while i < n and u[i] <= curr_x:
            i += 1
        while j < m and v[j] <= curr_x:
            j += 1
        if i >= n and j >= m:
            break
        if i < n and j < m:
            next_x = u[i] if u[i] < v[j] else v[j]
        elif i < n:
            next_x = u[i]
        else:
            next_x = v[j]
        dist += abs(float(i) * inv_n - float(j) * inv_m) * (next_x - curr_x)
        curr_x = next_x

    return dist


@njit(cache=True, nogil=True, parallel=True)
def _batch_spectral_distance_matrix_kernel(
    node_offsets: np.ndarray,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    edge_offsets: np.ndarray,
) -> np.ndarray:
    """Compute per-graph normalized Laplacian spectra and pairwise 1-Wasserstein distance matrix in parallel."""
    n_graphs = len(node_offsets) - 1
    dist_matrix = np.zeros((n_graphs, n_graphs), dtype=np.float64)
    if n_graphs == 0:
        return dist_matrix

    total_nodes = int(node_offsets[n_graphs])
    all_spectra = np.zeros(total_nodes, dtype=np.float64)

    for g in prange(n_graphs):
        m_start = int(node_offsets[g])
        m_end = int(node_offsets[g + 1])
        n_v = m_end - m_start
        if n_v == 0:
            continue

        lap = np.zeros((n_v, n_v), dtype=np.float64)
        e_start = int(edge_offsets[g])
        e_end = int(edge_offsets[g + 1])
        for k in range(e_start, e_end):
            u = int(edge_sources[k]) - m_start
            v = int(edge_targets[k]) - m_start
            if 0 <= u < n_v and 0 <= v < n_v and u != v:
                lap[u, v] = 1.0
                lap[v, u] = 1.0

        deg = np.zeros(n_v, dtype=np.float64)
        inv_sqrt = np.zeros(n_v, dtype=np.float64)
        for u in range(n_v):
            s = 0.0
            for v in range(n_v):
                s += lap[u, v]
            deg[u] = s
            if s > 0.0:
                inv_sqrt[u] = 1.0 / np.sqrt(s)

        for u in range(n_v):
            du = inv_sqrt[u]
            for v in range(n_v):
                if u == v:
                    lap[u, u] = 1.0 if deg[u] > 0.0 else 0.0
                elif lap[u, v] != 0.0:
                    lap[u, v] = -lap[u, v] * du * inv_sqrt[v]

        eigs = np.linalg.eigvalsh(lap)
        for u in range(n_v):
            all_spectra[m_start + u] = eigs[u]

    for i in prange(n_graphs):
        s_i = int(node_offsets[i])
        e_i = int(node_offsets[i + 1])
        spec_i = all_spectra[s_i:e_i]
        for j in range(i + 1, n_graphs):
            s_j = int(node_offsets[j])
            e_j = int(node_offsets[j + 1])
            d = _wasserstein_1d_sorted_kernel(spec_i, all_spectra[s_j:e_j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


@njit(cache=True, nogil=True, parallel=True)
def _slice_graph_batch_kernel(
    kept_indices: np.ndarray,
    node_types: np.ndarray,
    node_offsets: np.ndarray,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    edge_types: np.ndarray,
    edge_offsets: np.ndarray,
    new_node_offsets: np.ndarray,
    new_edge_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gather nodes and rebase edges in parallel across selected graphs in a `GraphBatch`."""
    num_kept = len(kept_indices)
    total_nodes = int(new_node_offsets[num_kept])
    total_edges = int(new_edge_offsets[num_kept])

    new_nodes = np.empty(total_nodes, dtype=np.uint32)
    new_sources = np.empty(total_edges, dtype=np.uint32)
    new_targets = np.empty(total_edges, dtype=np.uint32)
    new_types = np.empty(total_edges, dtype=np.uint32)

    for out_i in prange(num_kept):
        orig_i = kept_indices[out_i]
        m_start = int(node_offsets[orig_i])
        m_end = int(node_offsets[orig_i + 1])
        n_nodes = m_end - m_start
        dst_m_start = int(new_node_offsets[out_i])

        for j in range(n_nodes):
            new_nodes[dst_m_start + j] = node_types[m_start + j]

        e_start = int(edge_offsets[orig_i])
        e_end = int(edge_offsets[orig_i + 1])
        n_edges = e_end - e_start
        dst_e_start = int(new_edge_offsets[out_i])
        local_shift = dst_m_start - m_start

        for j in range(n_edges):
            new_sources[dst_e_start + j] = np.uint32(int(edge_sources[e_start + j]) + local_shift)
            new_targets[dst_e_start + j] = np.uint32(int(edge_targets[e_start + j]) + local_shift)
            new_types[dst_e_start + j] = edge_types[e_start + j]

    return new_nodes, new_sources, new_targets, new_types


@njit(cache=True, nogil=True)
def _erdos_renyi_edges_kernel(
    node_offsets: np.ndarray,
    p: float,
    n_edge_types: int,
    directed: bool,
    allow_self_loops: bool,
    seed: np.uint64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-pass deterministic Erdős–Rényi edge generator across a ragged batch of graphs."""
    n_graphs = len(node_offsets) - 1
    edge_offsets = np.zeros(n_graphs + 1, dtype=np.uint32)
    inv_u64 = 1.0 / 18446744073709551616.0  # 2^-64

    # Pass 1: count edges per graph using deterministic xorshift64* stream
    state = seed if seed != np.uint64(0) else np.uint64(0x9E3779B97F4A7C15)
    total_edges = 0
    for g in range(n_graphs):
        start = int(node_offsets[g])
        end = int(node_offsets[g + 1])
        if directed:
            for u in range(start, end):
                for v in range(start, end):
                    if not allow_self_loops and u == v:
                        continue
                    state ^= state >> np.uint64(12)
                    state ^= state << np.uint64(25)
                    state ^= state >> np.uint64(27)
                    rnd = state * np.uint64(2685821657736338717)
                    if float(rnd) * inv_u64 < p:
                        state ^= state >> np.uint64(12)
                        state ^= state << np.uint64(25)
                        state ^= state >> np.uint64(27)
                        total_edges += 1
        else:
            for u in range(start, end):
                v_start = u if allow_self_loops else u + 1
                for v in range(v_start, end):
                    state ^= state >> np.uint64(12)
                    state ^= state << np.uint64(25)
                    state ^= state >> np.uint64(27)
                    rnd = state * np.uint64(2685821657736338717)
                    if float(rnd) * inv_u64 < p:
                        state ^= state >> np.uint64(12)
                        state ^= state << np.uint64(25)
                        state ^= state >> np.uint64(27)
                        total_edges += 1 if u == v else 2
        edge_offsets[g + 1] = np.uint32(total_edges)

    sources = np.empty(total_edges, dtype=np.uint32)
    targets = np.empty(total_edges, dtype=np.uint32)
    types = np.empty(total_edges, dtype=np.uint32)

    if total_edges == 0:
        return sources, targets, types, edge_offsets

    # Pass 2: replay identical PRNG stream to populate edges without intermediate buffers
    state = seed if seed != np.uint64(0) else np.uint64(0x9E3779B97F4A7C15)
    cursor = 0
    n_et = np.uint64(n_edge_types)
    for g in range(n_graphs):
        start = int(node_offsets[g])
        end = int(node_offsets[g + 1])
        if directed:
            for u in range(start, end):
                for v in range(start, end):
                    if not allow_self_loops and u == v:
                        continue
                    state ^= state >> np.uint64(12)
                    state ^= state << np.uint64(25)
                    state ^= state >> np.uint64(27)
                    rnd = state * np.uint64(2685821657736338717)
                    if float(rnd) * inv_u64 < p:
                        state ^= state >> np.uint64(12)
                        state ^= state << np.uint64(25)
                        state ^= state >> np.uint64(27)
                        et = np.uint32(((state * np.uint64(2685821657736338717)) % n_et) + np.uint64(1))
                        sources[cursor] = np.uint32(u)
                        targets[cursor] = np.uint32(v)
                        types[cursor] = et
                        cursor += 1
        else:
            for u in range(start, end):
                v_start = u if allow_self_loops else u + 1
                for v in range(v_start, end):
                    state ^= state >> np.uint64(12)
                    state ^= state << np.uint64(25)
                    state ^= state >> np.uint64(27)
                    rnd = state * np.uint64(2685821657736338717)
                    if float(rnd) * inv_u64 < p:
                        state ^= state >> np.uint64(12)
                        state ^= state << np.uint64(25)
                        state ^= state >> np.uint64(27)
                        et = np.uint32(((state * np.uint64(2685821657736338717)) % n_et) + np.uint64(1))
                        sources[cursor] = np.uint32(u)
                        targets[cursor] = np.uint32(v)
                        types[cursor] = et
                        cursor += 1
                        if u != v:
                            sources[cursor] = np.uint32(v)
                            targets[cursor] = np.uint32(u)
                            types[cursor] = et
                            cursor += 1

    return sources, targets, types, edge_offsets
