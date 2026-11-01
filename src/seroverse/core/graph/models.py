"""Core graph representations, bitfield packing, and vectorized SoA containers."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields
from functools import cache
from typing import Any, ClassVar, Self

import numpy as np
from numba import prange

from seroverse.core.collections import Batch, View
from seroverse.core.graph.csr import CSR, RelationalCSR
from seroverse.core.types import Types

# Constants ------------------------------------------------------------------------------------------------------------
TAG_EDGE: int = 1 << 31  # 0x80000000 (Bit 31 discriminator for injected edge nodes)
_EMPTY_U32: Types.UInt32Array = np.empty(0, dtype=np.uint32)
_EMPTY_U32.setflags(write=False)


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
class Graph(View):
    """
    Scalar view and topological model representing a single directed graph.
    Provides sparse CSR conversion, multi-hop [`RelationalCSR`][seroverse.core.graph.csr.RelationalCSR]
    construction, Laplacian spectrum calculation, 1-Wasserstein spectral distance,
    subgraph extraction, and bitmask-aware VF2 subgraph/graph isomorphism.
    """

    _node_cls: ClassVar[type[Any]] = int
    _edge_cls: ClassVar[type[Any]] = DirectedEdge
    _edge_type_cls: ClassVar[type[Any]] = int

    nodes: tuple[Any, ...] = ()
    edges: tuple[Any, ...] = ()

    @cache  # noqa: B019
    def _packed_topology(
        self,
    ) -> tuple[
        Types.UInt32Array,
        Types.UInt32Array,
        Types.UInt32Array,
        Types.UInt32Array,
    ]:
        """Return cached contiguous `uint32` arrays `(node_types, edge_sources, edge_targets, edge_types)`."""
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        if n_nodes == 0:
            return _EMPTY_U32, _EMPTY_U32, _EMPTY_U32, _EMPTY_U32
        node_types = np.fromiter(
            (n.type_id if hasattr(n, "type_id") else int(n) for n in self.nodes),
            dtype=np.uint32,
            count=n_nodes,
        )
        node_types.setflags(write=False)
        if n_edges == 0:
            return node_types, _EMPTY_U32, _EMPTY_U32, _EMPTY_U32
        sources = np.fromiter((e.source for e in self.edges), dtype=np.uint32, count=n_edges)
        targets = np.fromiter((e.target for e in self.edges), dtype=np.uint32, count=n_edges)
        edge_types = np.fromiter((int(e.edge_type) for e in self.edges), dtype=np.uint32, count=n_edges)
        sources.setflags(write=False)
        targets.setflags(write=False)
        edge_types.setflags(write=False)
        return node_types, sources, targets, edge_types

    def in_degrees(self) -> Types.Int32Array:
        """Compute the 1D `int32` array of incoming edge counts per node."""
        n = len(self.nodes)
        deg = np.zeros(n, dtype=np.int32)
        for e in self.edges:
            if 0 <= e.target < n:
                deg[e.target] += 1
        return deg

    def out_degrees(self) -> Types.Int32Array:
        """Compute the 1D `int32` array of outgoing edge counts per node."""
        n = len(self.nodes)
        deg = np.zeros(n, dtype=np.int32)
        for e in self.edges:
            if 0 <= e.source < n:
                deg[e.source] += 1
        return deg

    def degrees(self) -> Types.Int32Array:
        """Compute the 1D `int32` array of total degree (`in_degrees + out_degrees`) per node."""
        n = len(self.nodes)
        deg = np.zeros(n, dtype=np.int32)
        for e in self.edges:
            if 0 <= e.source < n:
                deg[e.source] += 1
            if 0 <= e.target < n:
                deg[e.target] += 1
        return deg

    def subgraph(self, keep_nodes: Iterable[int] | Types.BoolArray) -> Self:
        """Extract the induced subgraph over `keep_nodes` (node index sequence or boolean mask) and rebase edges."""
        n = len(self.nodes)
        if isinstance(keep_nodes, np.ndarray) and keep_nodes.dtype == np.bool_:
            kept = [i for i, flag in enumerate(keep_nodes) if flag]
        else:
            kept = [int(i) for i in keep_nodes if 0 <= int(i) < n]

        if len(kept) == n and all(idx == i for i, idx in enumerate(kept)):
            return self

        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(kept)}
        edge_cls = type(self)._edge_cls
        new_nodes = tuple(self.nodes[old_idx] for old_idx in kept)
        new_edges = tuple(
            edge_cls(source=remap[e.source], target=remap[e.target], edge_type=e.edge_type)
            for e in self.edges
            if e.source in remap and e.target in remap
        )
        return self.replace(nodes=new_nodes, edges=new_edges)

    def filter_nodes(self, predicate: Callable[[Any], bool]) -> Self:
        """Return the induced subgraph retaining only nodes where `predicate(node)` is True."""
        return self.subgraph(i for i, node in enumerate(self.nodes) if predicate(node))

    def filter_edges(self, predicate: Callable[[Any], bool]) -> Self:
        """Return a copy of this graph retaining only edges where `predicate(edge)` is True."""
        new_edges = tuple(e for e in self.edges if predicate(e))
        if len(new_edges) == len(self.edges):
            return self
        return self.replace(edges=new_edges)

    def bfs_depths(self, root: int = 0, *, reverse: bool = False) -> dict[int, int]:
        """Compute shortest-path hop depths from `root` along directed edges (or reversed edges when `reverse=True`)."""
        n = len(self.nodes)
        if not (0 <= root < n):
            return {}
        adj: list[list[int]] = [[] for _ in range(n)]
        for e in self.edges:
            if 0 <= e.source < n and 0 <= e.target < n and e.source != e.target:
                if reverse:
                    adj[e.target].append(e.source)
                else:
                    adj[e.source].append(e.target)
        depths: dict[int, int] = {root: 0}
        queue: list[int] = [root]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            d_u = depths[u]
            for v in adj[u]:
                if v not in depths:
                    depths[v] = d_u + 1
                    queue.append(v)
        return depths

    def find_subgraph_mapping(
        self,
        other: Graph,
        *,
        node_mask: int = 0xFFFFFFFF,
        node_wildcard_masks: Sequence[int] | Types.UInt32Array = (),
        edge_mask: int = 0xFFFFFFFF,
        edge_wildcard_masks: Sequence[int] | Types.UInt32Array = (),
        require_bijection: bool = False,
        undirected: bool = False,
    ) -> dict[int, int] | None:
        """Find an injective edge-preserving homomorphism (`self` -> `other`) using compiled bitmask VF2.

        Args:
            other: Target supergraph to match `self` into.
            node_mask: Strict 32-bit bitmask that must match identically between `self` and `other` nodes.
            node_wildcard_masks: Sequence of 32-bit bitfield masks where `0` acts as a wildcard.
            edge_mask: Strict 32-bit bitmask that must match identically between `self` and `other` edge types.
            edge_wildcard_masks: Sequence of 32-bit edge bitfield masks where `0` acts as a wildcard.
            require_bijection: If True, requires a full graph isomorphism (`len(self) == len(other)` and equal edge counts).
            undirected: If True, treats edges as symmetric undirected bonds during matching.

        Returns:
            Mapping `{u_sub: v_sup}` from 0-based node indices in `self` to `other`, or `None` if no match exists.
        """
        sub_n, sub_s, sub_t, sub_et = self._packed_topology()
        sup_n, sup_s, sup_t, sup_et = other._packed_topology()
        nw_arr = (
            node_wildcard_masks
            if isinstance(node_wildcard_masks, np.ndarray) and node_wildcard_masks.dtype == np.uint32
            else (np.asarray(node_wildcard_masks, dtype=np.uint32) if len(node_wildcard_masks) > 0 else _EMPTY_U32)
        )
        ew_arr = (
            edge_wildcard_masks
            if isinstance(edge_wildcard_masks, np.ndarray) and edge_wildcard_masks.dtype == np.uint32
            else (np.asarray(edge_wildcard_masks, dtype=np.uint32) if len(edge_wildcard_masks) > 0 else _EMPTY_U32)
        )
        found, mapping = _vf2_match_kernel(
            sub_n,
            sub_s,
            sub_t,
            sub_et,
            sup_n,
            sup_s,
            sup_t,
            sup_et,
            np.uint32(node_mask & 0xFFFFFFFF),
            nw_arr,
            np.uint32(edge_mask & 0xFFFFFFFF),
            ew_arr,
            bool(require_bijection),
            bool(undirected),
        )
        if not found:
            return None
        return {u: int(mapping[u]) for u in range(len(sub_n))}

    def match_subgraph(
        self,
        other: Graph,
        *,
        node_mask: int = 0xFFFFFFFF,
        node_wildcard_masks: Sequence[int] | Types.UInt32Array = (),
        edge_mask: int = 0xFFFFFFFF,
        edge_wildcard_masks: Sequence[int] | Types.UInt32Array = (),
        require_bijection: bool = False,
        undirected: bool = False,
    ) -> bool:
        """Return True if an injective edge-preserving homomorphism (`self` -> `other`) exists."""
        sub_n, sub_s, sub_t, sub_et = self._packed_topology()
        sup_n, sup_s, sup_t, sup_et = other._packed_topology()
        nw_arr = (
            node_wildcard_masks
            if isinstance(node_wildcard_masks, np.ndarray) and node_wildcard_masks.dtype == np.uint32
            else (np.asarray(node_wildcard_masks, dtype=np.uint32) if len(node_wildcard_masks) > 0 else _EMPTY_U32)
        )
        ew_arr = (
            edge_wildcard_masks
            if isinstance(edge_wildcard_masks, np.ndarray) and edge_wildcard_masks.dtype == np.uint32
            else (np.asarray(edge_wildcard_masks, dtype=np.uint32) if len(edge_wildcard_masks) > 0 else _EMPTY_U32)
        )
        found, _ = _vf2_match_kernel(
            sub_n,
            sub_s,
            sub_t,
            sub_et,
            sup_n,
            sup_s,
            sup_t,
            sup_et,
            np.uint32(node_mask & 0xFFFFFFFF),
            nw_arr,
            np.uint32(edge_mask & 0xFFFFFFFF),
            ew_arr,
            bool(require_bijection),
            bool(undirected),
        )
        return bool(found)

    @cache  # noqa: B019
    def as_graph(
        self,
        as_bipartite: bool = False,
        symmetrize: bool = False,
    ) -> tuple[CSR, np.ndarray[tuple[int], np.dtype[np.uint32]]]:
        """
        Builds a sparse CSR adjacency matrix and node label array.

        All biological and genomic models in SeroVerse are strictly directed graphs.
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
            other: Another [`Graph`][seroverse.core.graph.models.Graph] instance to compare against.

        Returns:
            Wasserstein distance between the two normalized Laplacian spectra as a float.
        """
        spec1 = self.laplacian_spectrum()
        spec2 = other.laplacian_spectrum()
        return float(_wasserstein_1d_sorted_kernel(spec1, spec2))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self.nodes)} nodes, {len(self.edges)} edges)"


@dataclass(slots=True, frozen=True)
class Graphs[T: Graph = Graph](Batch[T]):
    """
    Base Structure-of-Arrays (SoA) container for batched graph datasets.
    Provides block-diagonal CSR adjacency, Wasserstein spectral distance matrices,
    pan-graph compression, and vectorized SoA array slicing and concatenation.
    """

    _CSR_FIELDS: ClassVar[tuple[str, ...]] = (
        "node_types",
        "node_offsets",
        "edge_sources",
        "edge_targets",
        "edge_types",
        "edge_offsets",
    )
    _record_fields: ClassVar[tuple[str, ...]] = ()
    _scalar_type: ClassVar[type[Any] | None] = Graph

    node_types: Types.UInt32Array
    node_offsets: Types.UInt32Array
    edge_sources: Types.UInt32Array
    edge_targets: Types.UInt32Array
    edge_types: Types.UInt32Array
    edge_offsets: Types.UInt32Array

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__dataclass_fields__" not in cls.__dict__:
            return
        cls._record_fields = tuple(f for f in cls._field_dtypes if f not in cls._CSR_FIELDS)

    def __len__(self) -> int:
        return len(self.node_offsets) - 1

    @property
    def total_nodes(self) -> int:
        return len(self.node_types)

    @property
    def total_edges(self) -> int:
        return len(self.edge_sources)

    @property
    def node_lengths(self) -> Types.UInt32Array:
        """Number of nodes in each graph in the batch."""
        return np.diff(self.node_offsets)

    @property
    def edge_lengths(self) -> Types.UInt32Array:
        """Number of edges in each graph in the batch."""
        return np.diff(self.edge_offsets)

    @classmethod
    def empty(cls) -> Self:
        """Create an empty 0-length graph batch."""
        kwargs: dict[str, Any] = {
            "node_types": np.empty(0, dtype=np.uint32),
            "node_offsets": np.zeros(1, dtype=np.uint32),
            "edge_sources": np.empty(0, dtype=np.uint32),
            "edge_targets": np.empty(0, dtype=np.uint32),
            "edge_types": np.empty(0, dtype=np.uint32),
            "edge_offsets": np.zeros(1, dtype=np.uint32),
        }
        for f in cls._record_fields:
            kwargs[f] = np.empty(0, dtype=cls._field_dtypes[f])
        for f, child_cls in cls._nested_fields.items():
            kwargs[f] = child_cls.empty()
        return cls(**kwargs)

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenate multiple graph batches into a single contiguous batch."""
        batch_list: list[Self] = [b for b in batches if len(b) > 0]
        if not batch_list:
            return cls.empty()
        if len(batch_list) == 1:
            return batch_list[0]

        nodes, n_offs, sources, targets, types, e_offs = cls._concat_soa_base(batch_list)
        kwargs: dict[str, Any] = {
            "node_types": nodes,
            "node_offsets": n_offs,
            "edge_sources": sources,
            "edge_targets": targets,
            "edge_types": types,
            "edge_offsets": e_offs,
        }
        for f in cls._record_fields:
            kwargs[f] = np.concatenate([getattr(b, f) for b in batch_list])
        for f, child_cls in cls._nested_fields.items():
            kwargs[f] = child_cls.concat([getattr(b, f) for b in batch_list])
        return cls(**kwargs)

    @classmethod
    def from_list(cls, items: Sequence[T]) -> Self:
        """Pack a sequence of [`Graph`][seroverse.core.graph.models.Graph] instances into a [`Graphs`][seroverse.core.graph.models.Graphs] batch."""
        n_graphs = len(items)
        if n_graphs == 0:
            return cls.empty()

        nodes_list = [g.nodes for g in items]
        edges_list = [g.edges for g in items]
        node_offsets = np.zeros(n_graphs + 1, dtype=np.uint32)
        edge_offsets = np.zeros(n_graphs + 1, dtype=np.uint32)
        tot_nodes = sum(len(ns) for ns in nodes_list)
        tot_edges = sum(len(es) for es in edges_list)

        node_types = np.empty(tot_nodes, dtype=np.uint32)
        edge_sources = np.empty(tot_edges, dtype=np.uint32)
        edge_targets = np.empty(tot_edges, dtype=np.uint32)
        edge_types = np.empty(tot_edges, dtype=np.uint32)

        curr_n = 0
        curr_e = 0
        for i, (ns, es) in enumerate(zip(nodes_list, edges_list, strict=True)):
            gn = len(ns)
            ge = len(es)
            for j, node in enumerate(ns):
                node_types[curr_n + j] = node.type_id if hasattr(node, "type_id") else int(node)
            for j, edge in enumerate(es):
                edge_sources[curr_e + j] = edge.source + curr_n
                edge_targets[curr_e + j] = edge.target + curr_n
                edge_types[curr_e + j] = int(edge.edge_type)
            curr_n += gn
            curr_e += ge
            node_offsets[i + 1] = curr_n
            edge_offsets[i + 1] = curr_e

        kwargs: dict[str, Any] = {
            "node_types": node_types,
            "node_offsets": node_offsets,
            "edge_sources": edge_sources,
            "edge_targets": edge_targets,
            "edge_types": edge_types,
            "edge_offsets": edge_offsets,
        }
        if cls._record_fields or cls._nested_fields:
            scalar_cls = cls._scalar_type or type(items[0])
            s_fields = tuple(f.name for f in fields(scalar_cls) if f.name not in ("nodes", "edges"))
            extra_b_fields = tuple(f for f in cls._all_fields if f not in cls._CSR_FIELDS)
            field_map = dict(zip(extra_b_fields, s_fields, strict=True))
            for b_f in cls._record_fields:
                s_f = field_map[b_f]
                kwargs[b_f] = np.array([getattr(it, s_f) for it in items], dtype=cls._field_dtypes[b_f])
            for b_f, child_cls in cls._nested_fields.items():
                s_f = field_map[b_f]
                kwargs[b_f] = child_cls.from_list([getattr(it, s_f) for it in items])
        return cls(**kwargs)

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
        r"""Generate a random [`Graphs`][seroverse.core.graph.models.Graphs] batch using the Erdős–Rényi $G(n, p)$ model.

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
            Self: A newly generated [`Graphs`][seroverse.core.graph.models.Graphs] instance.
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

    def _get_scalar(self, idx: int) -> T:
        """Materialize the [`Graph`][seroverse.core.graph.models.Graph] (or subclass `T`) at index `idx`."""
        n_s, n_e = int(self.node_offsets[idx]), int(self.node_offsets[idx + 1])
        e_s, e_e = int(self.edge_offsets[idx]), int(self.edge_offsets[idx + 1])
        scalar_cls: type[Any] = self._scalar_type or Graph
        node_cls = getattr(scalar_cls, "_node_cls", int)
        edge_cls = getattr(scalar_cls, "_edge_cls", DirectedEdge)
        edge_type_cls = getattr(scalar_cls, "_edge_type_cls", int)
        nodes = tuple(node_cls(int(x)) for x in self.node_types[n_s:n_e])
        edges = tuple(
            edge_cls(
                source=int(self.edge_sources[k]) - n_s,
                target=int(self.edge_targets[k]) - n_s,
                edge_type=edge_type_cls(int(self.edge_types[k])),
            )
            for k in range(e_s, e_e)
        )
        if not self._record_fields and not self._nested_fields:
            return scalar_cls(nodes=nodes, edges=edges)
        extra_b_fields = tuple(f for f in self._all_fields if f not in self._CSR_FIELDS)
        s_fields = tuple(f.name for f in fields(scalar_cls) if f.name not in ("nodes", "edges"))
        kwargs: dict[str, Any] = {"nodes": nodes, "edges": edges}
        for b_f, s_f in zip(extra_b_fields, s_fields, strict=True):
            val = getattr(self, b_f)[idx]
            if b_f in self._field_dtypes:
                dt = self._field_dtypes[b_f]
                if dt.kind in ("S", "a"):
                    val = bytes(val)
                elif hasattr(val, "item"):
                    val = val.item()
            kwargs[s_f] = val
        return scalar_cls(**kwargs)

    def _get_slice(self, item: slice | np.ndarray[Any, Any] | list[Any]) -> Self:
        """Slice a sub-batch of graphs by slice, boolean mask, or index array."""
        indices = self._normalize_indices(item)
        if len(indices) == 0:
            return self.empty()
        nodes_arr, n_offs, sources, targets, types, e_offs = self._slice_soa_arrays(indices)
        kwargs: dict[str, Any] = {
            "node_types": nodes_arr,
            "node_offsets": n_offs,
            "edge_sources": sources,
            "edge_targets": targets,
            "edge_types": types,
            "edge_offsets": e_offs,
        }
        for f in self._record_fields:
            kwargs[f] = getattr(self, f)[indices]
        for f in self._nested_fields:
            kwargs[f] = getattr(self, f)[indices]
        return type(self)(**kwargs)

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

    def wwl_distance_matrix(
        self,
        node_features: np.ndarray[tuple[int, int], np.dtype[Any]],
        iterations: int = 2,
        smoothing_factor: float = 0.5,
    ) -> Types.Float64Array2D:
        """Compute the all-vs-all Wasserstein Weisfeiler-Lehman (WWL) graph distance matrix across the batch.

        Topologically smooths initial `node_features` of shape `(total_nodes, D)` over symmetrized
        neighborhoods for `iterations` steps and computes the exact ND 1-Wasserstein (Earth Mover's)
        distance between resulting node embedding point clouds in parallel via compiled Numba kernels.

        Args:
            node_features: 2D array of shape `(total_nodes, D)` containing initial continuous node features.
            iterations: Number of Weisfeiler-Lehman neighborhood smoothing iterations (default `2`).
            smoothing_factor: Neighborhood averaging weight in `[0.0, 1.0]` (default `0.5`).

        Returns:
            Symmetric `(len(self), len(self))` `float64` distance matrix.
        """
        n_graphs = len(self)
        n_nodes = len(self.node_types)
        if n_graphs == 0 or n_nodes == 0:
            return np.zeros((n_graphs, n_graphs), dtype=np.float64)
        feats = np.ascontiguousarray(node_features, dtype=np.float64)
        smoothed = _wl_smooth_embeddings_kernel(
            feats,
            self.node_offsets,
            self.edge_sources,
            self.edge_targets,
            self.edge_offsets,
            int(iterations),
            float(smoothing_factor),
        )
        return _batch_wwl_distance_kernel(smoothed, self.node_offsets)

    def match_subgraph_batch(
        self,
        query: Graph,
        *,
        node_mask: int = 0xFFFFFFFF,
        node_wildcard_masks: Sequence[int] | Types.UInt32Array = (),
        edge_mask: int = 0xFFFFFFFF,
        edge_wildcard_masks: Sequence[int] | Types.UInt32Array = (),
        require_bijection: bool = False,
        undirected: bool = False,
    ) -> Types.BoolArray:
        """Screen all graphs in this batch in parallel for a subgraph/isomorphism match with `query`."""
        n_graphs = len(self)
        if n_graphs == 0:
            return np.empty(0, dtype=np.bool_)
        sub_n, sub_s, sub_t, sub_et = query._packed_topology()
        nw_arr = (
            node_wildcard_masks
            if isinstance(node_wildcard_masks, np.ndarray) and node_wildcard_masks.dtype == np.uint32
            else (np.asarray(node_wildcard_masks, dtype=np.uint32) if len(node_wildcard_masks) > 0 else _EMPTY_U32)
        )
        ew_arr = (
            edge_wildcard_masks
            if isinstance(edge_wildcard_masks, np.ndarray) and edge_wildcard_masks.dtype == np.uint32
            else (np.asarray(edge_wildcard_masks, dtype=np.uint32) if len(edge_wildcard_masks) > 0 else _EMPTY_U32)
        )
        return _batch_vf2_has_match_kernel(
            sub_n,
            sub_s,
            sub_t,
            sub_et,
            self.node_types,
            self.node_offsets,
            self.edge_sources,
            self.edge_targets,
            self.edge_types,
            self.edge_offsets,
            np.uint32(node_mask & 0xFFFFFFFF),
            nw_arr,
            np.uint32(edge_mask & 0xFFFFFFFF),
            ew_arr,
            bool(require_bijection),
            bool(undirected),
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

        Pan-graphs in SeroVerse are strictly directed consensus networks representing
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
@Types.njit
def _wasserstein_1d_sorted_kernel(u: Types.Float64Array, v: Types.Float64Array) -> float:
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


@Types.njit
def _sym_eigvalsh_kernel(a: Types.Float64Array2D) -> Types.Float64Array:
    """Compute sorted eigenvalues of a real symmetric matrix in-place via cyclic Jacobi rotations."""
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    if n == 1:
        out[0] = a[0, 0]
        return out

    for _ in range(50):
        max_off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = abs(a[p, q])
                if apq > max_off:
                    max_off = apq
        if max_off < 1e-13:
            break

        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p, q]
                if abs(apq) < 1e-15:
                    continue
                diff = a[q, q] - a[p, p]
                if abs(apq) <= 1e-15 * abs(diff):
                    t = apq / diff
                else:
                    tau = diff / (2.0 * apq)
                    t = (1.0 if tau >= 0.0 else -1.0) / (abs(tau) + np.sqrt(1.0 + tau * tau))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = t * c
                tau_rot = s / (1.0 + c)
                a[p, p] -= t * apq
                a[q, q] += t * apq
                a[p, q] = 0.0
                a[q, p] = 0.0
                for r in range(n):
                    if r != p and r != q:
                        arp = a[r, p]
                        arq = a[r, q]
                        new_rp = arp - s * (arq + tau_rot * arp)
                        new_rq = arq + s * (arp - tau_rot * arq)
                        a[r, p] = new_rp
                        a[p, r] = new_rp
                        a[r, q] = new_rq
                        a[q, r] = new_rq

    for i in range(n):
        out[i] = a[i, i]
    out.sort()
    return out


@Types.njit(parallel=True)
def _batch_spectral_distance_matrix_kernel(
    node_offsets: Types.UInt32Array,
    edge_sources: Types.UInt32Array,
    edge_targets: Types.UInt32Array,
    edge_offsets: Types.UInt32Array,
) -> Types.Float64Array2D:
    """Compute per-graph normalized Laplacian spectra and pairwise 1-Wasserstein distance matrix in parallel."""
    n_graphs = len(node_offsets) - 1
    dist_matrix = np.zeros((n_graphs, n_graphs), dtype=np.float64)
    if n_graphs == 0:
        return dist_matrix

    total_nodes = int(node_offsets[n_graphs])
    all_spectra = np.zeros(total_nodes, dtype=np.float64)

    for g in prange(n_graphs):  # type: ignore[not-iterable]
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

        eigs = _sym_eigvalsh_kernel(lap)
        for u in range(n_v):
            all_spectra[m_start + u] = eigs[u]

    for i in prange(n_graphs):  # type: ignore[not-iterable]
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


@Types.njit(parallel=True)
def _slice_graph_batch_kernel(
    kept_indices: Types.Int64Array,
    node_types: Types.UInt32Array,
    node_offsets: Types.UInt32Array,
    edge_sources: Types.UInt32Array,
    edge_targets: Types.UInt32Array,
    edge_types: Types.UInt32Array,
    edge_offsets: Types.UInt32Array,
    new_node_offsets: Types.UInt32Array,
    new_edge_offsets: Types.UInt32Array,
) -> tuple[Types.UInt32Array, Types.UInt32Array, Types.UInt32Array, Types.UInt32Array]:
    """Gather nodes and rebase edges in parallel across selected graphs in a `Graphs` batch."""
    num_kept = len(kept_indices)
    total_nodes = int(new_node_offsets[num_kept])
    total_edges = int(new_edge_offsets[num_kept])

    new_nodes = np.empty(total_nodes, dtype=np.uint32)
    new_sources = np.empty(total_edges, dtype=np.uint32)
    new_targets = np.empty(total_edges, dtype=np.uint32)
    new_types = np.empty(total_edges, dtype=np.uint32)

    for out_i in prange(num_kept):  # type: ignore[not-iterable]
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


@Types.njit
def _erdos_renyi_edges_kernel(
    node_offsets: Types.UInt32Array,
    p: float,
    n_edge_types: int,
    directed: bool,
    allow_self_loops: bool,
    seed: np.uint64,
) -> tuple[Types.UInt32Array, Types.UInt32Array, Types.UInt32Array, Types.UInt32Array]:
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


@Types.njit(inline="always")
def _bitmask_compat_kernel(
    u_val: np.uint32,
    v_val: np.uint32,
    strict_mask: np.uint32,
    wildcard_masks: Types.UInt32Array,
) -> bool:
    """Return True if 32-bit packed integers `u_val` and `v_val` satisfy strict and wildcard bitmasks."""
    if (u_val & strict_mask) != (v_val & strict_mask):
        return False
    for i in range(len(wildcard_masks)):
        m = wildcard_masks[i]
        u_f = u_val & m
        v_f = v_val & m
        if u_f != v_f and u_f != np.uint32(0) and v_f != np.uint32(0):
            return False
    return True


@Types.njit(inline="always")
def _pair_edges_compat_kernel(
    u_start: int,
    u_end: int,
    w_target: int,
    sub_nbrs: Types.Int32Array,
    sub_etypes: Types.UInt32Array,
    v_start: int,
    v_end: int,
    vw_target: int,
    sup_nbrs: Types.Int32Array,
    sup_etypes: Types.UInt32Array,
    edge_mask: np.uint32,
    edge_wildcard_masks: Types.UInt32Array,
) -> bool:
    """Check if all parallel directed edges `u -> w_target` can be injectively matched into `v -> vw_target`."""
    req_cnt = 0
    for idx_u in range(u_start, u_end):
        if sub_nbrs[idx_u] == w_target:
            req_cnt += 1
    if req_cnt == 0:
        return True

    avail_cnt = 0
    for idx_v in range(v_start, v_end):
        if sup_nbrs[idx_v] == vw_target:
            avail_cnt += 1
    if req_cnt > avail_cnt:
        return False

    if req_cnt == 1:
        for idx_u in range(u_start, u_end):
            if sub_nbrs[idx_u] == w_target:
                rl = sub_etypes[idx_u]
                for idx_v in range(v_start, v_end):
                    if sup_nbrs[idx_v] == vw_target and _bitmask_compat_kernel(
                        rl, sup_etypes[idx_v], edge_mask, edge_wildcard_masks
                    ):
                        return True
                return False

    used_bits = np.uint64(0)
    for idx_u in range(u_start, u_end):
        if sub_nbrs[idx_u] != w_target:
            continue
        rl = sub_etypes[idx_u]
        matched = False
        local_idx = 0
        for idx_v in range(v_start, v_end):
            if sup_nbrs[idx_v] != vw_target:
                continue
            bit = np.uint64(1) << np.uint64(local_idx)
            if (used_bits & bit) == np.uint64(0) and _bitmask_compat_kernel(
                rl, sup_etypes[idx_v], edge_mask, edge_wildcard_masks
            ):
                used_bits |= bit
                matched = True
                break
            local_idx += 1
        if not matched:
            return False
    return True


@Types.njit
def _build_adj_lists_kernel(
    n_nodes: int,
    sources: Types.UInt32Array,
    targets: Types.UInt32Array,
    edge_types: Types.UInt32Array,
    undirected: bool,
) -> tuple[
    Types.Int32Array,
    Types.Int32Array,
    Types.UInt32Array,
    Types.Int32Array,
    Types.Int32Array,
    Types.UInt32Array,
]:
    """Construct flat CSR outgoing and incoming adjacency lists for VF2 matching."""
    out_deg = np.zeros(n_nodes, dtype=np.int32)
    in_deg = np.zeros(n_nodes, dtype=np.int32)
    n_edges = len(sources)

    for k in range(n_edges):
        u = int(sources[k])
        v = int(targets[k])
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            out_deg[u] += 1
            in_deg[v] += 1
            if undirected and u != v:
                out_deg[v] += 1
                in_deg[u] += 1

    out_indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    in_indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    for i in range(n_nodes):
        out_indptr[i + 1] = out_indptr[i] + out_deg[i]
        in_indptr[i + 1] = in_indptr[i] + in_deg[i]

    tot_out = int(out_indptr[n_nodes])
    tot_in = int(in_indptr[n_nodes])
    out_nbrs = np.empty(tot_out, dtype=np.int32)
    out_etypes = np.empty(tot_out, dtype=np.uint32)
    in_nbrs = np.empty(tot_in, dtype=np.int32)
    in_etypes = np.empty(tot_in, dtype=np.uint32)

    out_cur = out_indptr[:-1].copy()
    in_cur = in_indptr[:-1].copy()

    for k in range(n_edges):
        u = int(sources[k])
        v = int(targets[k])
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            et = edge_types[k]
            po = out_cur[u]
            out_nbrs[po] = v
            out_etypes[po] = et
            out_cur[u] = po + 1

            pi = in_cur[v]
            in_nbrs[pi] = u
            in_etypes[pi] = et
            in_cur[v] = pi + 1

            if undirected and u != v:
                po2 = out_cur[v]
                out_nbrs[po2] = u
                out_etypes[po2] = et
                out_cur[v] = po2 + 1

                pi2 = in_cur[u]
                in_nbrs[pi2] = v
                in_etypes[pi2] = et
                in_cur[u] = pi2 + 1

    return out_indptr, out_nbrs, out_etypes, in_indptr, in_nbrs, in_etypes


@Types.njit
def _vf2_match_kernel(
    sub_nodes: Types.UInt32Array,
    sub_sources: Types.UInt32Array,
    sub_targets: Types.UInt32Array,
    sub_edge_types: Types.UInt32Array,
    sup_nodes: Types.UInt32Array,
    sup_sources: Types.UInt32Array,
    sup_targets: Types.UInt32Array,
    sup_edge_types: Types.UInt32Array,
    node_mask: np.uint32,
    node_wildcard_masks: Types.UInt32Array,
    edge_mask: np.uint32,
    edge_wildcard_masks: Types.UInt32Array,
    require_bijection: bool,
    undirected: bool,
) -> tuple[bool, Types.Int32Array]:
    """Compiled bitmask-aware VF2 subgraph/graph isomorphism solver."""
    n_sub = len(sub_nodes)
    n_sup = len(sup_nodes)
    e_sub = len(sub_sources)
    e_sup = len(sup_sources)

    empty_map = np.empty(0, dtype=np.int32)
    if require_bijection and (n_sub != n_sup or e_sub != e_sup):
        return False, empty_map
    if n_sub > n_sup or e_sub > e_sup:
        return False, empty_map
    if n_sub == 0:
        return True, empty_map

    # Fast multiset pre-check when no node wildcards are active
    if len(node_wildcard_masks) == 0 and node_mask != np.uint32(0):
        sub_masked = np.empty(n_sub, dtype=np.uint32)
        sup_masked = np.empty(n_sup, dtype=np.uint32)
        for i in range(n_sub):
            sub_masked[i] = sub_nodes[i] & node_mask
        for i in range(n_sup):
            sup_masked[i] = sup_nodes[i] & node_mask
        sub_masked.sort()
        sup_masked.sort()
        if require_bijection:
            for i in range(n_sub):
                if sub_masked[i] != sup_masked[i]:
                    return False, empty_map
        else:
            j_sup = 0
            for i in range(n_sub):
                val = sub_masked[i]
                while j_sup < n_sup and sup_masked[j_sup] < val:
                    j_sup += 1
                if j_sup >= n_sup or sup_masked[j_sup] != val:
                    return False, empty_map
                j_sup += 1

    out_sub_p, out_sub_n, out_sub_e, in_sub_p, in_sub_n, in_sub_e = _build_adj_lists_kernel(
        n_sub, sub_sources, sub_targets, sub_edge_types, undirected
    )
    out_sup_p, out_sup_n, out_sup_e, in_sup_p, in_sup_n, in_sup_e = _build_adj_lists_kernel(
        n_sup, sup_sources, sup_targets, sup_edge_types, undirected
    )

    cand_mat = np.zeros((n_sub, n_sup), dtype=np.bool_)
    cand_counts = np.zeros(n_sub, dtype=np.int32)

    for u in range(n_sub):
        u_val = sub_nodes[u]
        uo_s, uo_e = int(out_sub_p[u]), int(out_sub_p[u + 1])
        ui_s, ui_e = int(in_sub_p[u]), int(in_sub_p[u + 1])
        u_out_deg = uo_e - uo_s
        u_in_deg = ui_e - ui_s
        cnt = 0

        for v in range(n_sup):
            if not _bitmask_compat_kernel(u_val, sup_nodes[v], node_mask, node_wildcard_masks):
                continue
            vo_s, vo_e = int(out_sup_p[v]), int(out_sup_p[v + 1])
            vi_s, vi_e = int(in_sup_p[v]), int(in_sup_p[v + 1])
            v_out_deg = vo_e - vo_s
            v_in_deg = vi_e - vi_s

            if require_bijection:
                if u_out_deg != v_out_deg or u_in_deg != v_in_deg:
                    continue
            else:
                if u_out_deg > v_out_deg or u_in_deg > v_in_deg:
                    continue

            # 1-hop outgoing lookahead
            ok = True
            for idx_u in range(uo_s, uo_e):
                w = out_sub_n[idx_u]
                w_val = sub_nodes[w]
                l_sub = out_sub_e[idx_u]
                found_nbr = False
                for idx_v in range(vo_s, vo_e):
                    vw = out_sup_n[idx_v]
                    if _bitmask_compat_kernel(
                        l_sub, out_sup_e[idx_v], edge_mask, edge_wildcard_masks
                    ) and _bitmask_compat_kernel(w_val, sup_nodes[vw], node_mask, node_wildcard_masks):
                        found_nbr = True
                        break
                if not found_nbr:
                    ok = False
                    break
            if not ok:
                continue

            # 1-hop incoming lookahead
            if not undirected:
                for idx_u in range(ui_s, ui_e):
                    w = in_sub_n[idx_u]
                    w_val = sub_nodes[w]
                    l_sub = in_sub_e[idx_u]
                    found_nbr = False
                    for idx_v in range(vi_s, vi_e):
                        vw = in_sup_n[idx_v]
                        if _bitmask_compat_kernel(
                            l_sub, in_sup_e[idx_v], edge_mask, edge_wildcard_masks
                        ) and _bitmask_compat_kernel(w_val, sup_nodes[vw], node_mask, node_wildcard_masks):
                            found_nbr = True
                            break
                    if not found_nbr:
                        ok = False
                        break
                if not ok:
                    continue

            cand_mat[u, v] = True
            cnt += 1

        if cnt == 0:
            return False, empty_map
        cand_counts[u] = cnt

    # Connected search ordering: start from most constrained node, then expand connected frontier
    order = np.empty(n_sub, dtype=np.int32)
    in_order = np.zeros(n_sub, dtype=np.bool_)
    in_frontier = np.zeros(n_sub, dtype=np.bool_)

    for step_idx in range(n_sub):
        has_front = False
        for u in range(n_sub):
            if in_frontier[u] and not in_order[u]:
                has_front = True
                break
        best_u = -1
        best_cands = 1 << 30
        best_deg = -1
        for u in range(n_sub):
            if in_order[u]:
                continue
            if has_front and not in_frontier[u]:
                continue
            c_u = int(cand_counts[u])
            d_u = int(out_sub_p[u + 1] - out_sub_p[u]) + int(in_sub_p[u + 1] - in_sub_p[u])
            if c_u < best_cands or (c_u == best_cands and d_u > best_deg):
                best_u = u
                best_cands = c_u
                best_deg = d_u

        order[step_idx] = best_u
        in_order[best_u] = True
        in_frontier[best_u] = False
        for idx_u in range(out_sub_p[best_u], out_sub_p[best_u + 1]):
            nb = out_sub_n[idx_u]
            if not in_order[nb]:
                in_frontier[nb] = True
        for idx_u in range(in_sub_p[best_u], in_sub_p[best_u + 1]):
            nb = in_sub_n[idx_u]
            if not in_order[nb]:
                in_frontier[nb] = True

    # Iterative stack-based VF2 backtracking
    mapping = np.full(n_sub, -1, dtype=np.int32)
    used_sup = np.zeros(n_sup, dtype=np.bool_)
    cand_cursor = np.zeros(n_sub, dtype=np.int32)
    step = 0
    steps_taken = 0
    max_steps = 8192

    while step >= 0:
        if step == n_sub:
            return True, mapping
        u = int(order[step])
        uo_s, uo_e = int(out_sub_p[u]), int(out_sub_p[u + 1])
        ui_s, ui_e = int(in_sub_p[u]), int(in_sub_p[u + 1])

        advanced = False
        for v in range(int(cand_cursor[step]), n_sup):
            if not cand_mat[u, v] or used_sup[v]:
                continue
            vo_s, vo_e = int(out_sup_p[v]), int(out_sup_p[v + 1])
            vi_s, vi_e = int(in_sup_p[v]), int(in_sup_p[v + 1])

            consistent = True
            for idx_u in range(uo_s, uo_e):
                w = int(out_sub_n[idx_u])
                if w == u:
                    if not _pair_edges_compat_kernel(
                        uo_s,
                        uo_e,
                        u,
                        out_sub_n,
                        out_sub_e,
                        vo_s,
                        vo_e,
                        v,
                        out_sup_n,
                        out_sup_e,
                        edge_mask,
                        edge_wildcard_masks,
                    ):
                        consistent = False
                        break
                elif mapping[w] != -1:
                    vw = int(mapping[w])
                    if not _pair_edges_compat_kernel(
                        uo_s,
                        uo_e,
                        w,
                        out_sub_n,
                        out_sub_e,
                        vo_s,
                        vo_e,
                        vw,
                        out_sup_n,
                        out_sup_e,
                        edge_mask,
                        edge_wildcard_masks,
                    ):
                        consistent = False
                        break
            if not consistent:
                continue

            if not undirected:
                for idx_u in range(ui_s, ui_e):
                    w = int(in_sub_n[idx_u])
                    if w != u and mapping[w] != -1:
                        vw = int(mapping[w])
                        if not _pair_edges_compat_kernel(
                            ui_s,
                            ui_e,
                            w,
                            in_sub_n,
                            in_sub_e,
                            vi_s,
                            vi_e,
                            vw,
                            in_sup_n,
                            in_sup_e,
                            edge_mask,
                            edge_wildcard_masks,
                        ):
                            consistent = False
                            break
                if not consistent:
                    continue

            cand_cursor[step] = v + 1
            mapping[u] = v
            used_sup[v] = True
            step += 1
            if step < n_sub:
                cand_cursor[step] = 0
            advanced = True
            break

        if not advanced:
            steps_taken += 1
            if steps_taken > max_steps:
                return False, empty_map
            step -= 1
            if step >= 0:
                prev_u = int(order[step])
                prev_v = int(mapping[prev_u])
                used_sup[prev_v] = False
                mapping[prev_u] = -1

    return False, empty_map


@Types.njit(parallel=True)
def _batch_vf2_has_match_kernel(
    sub_nodes: Types.UInt32Array,
    sub_sources: Types.UInt32Array,
    sub_targets: Types.UInt32Array,
    sub_edge_types: Types.UInt32Array,
    batch_node_types: Types.UInt32Array,
    batch_node_offsets: Types.UInt32Array,
    batch_edge_sources: Types.UInt32Array,
    batch_edge_targets: Types.UInt32Array,
    batch_edge_types: Types.UInt32Array,
    batch_edge_offsets: Types.UInt32Array,
    node_mask: np.uint32,
    node_wildcard_masks: Types.UInt32Array,
    edge_mask: np.uint32,
    edge_wildcard_masks: Types.UInt32Array,
    require_bijection: bool,
    undirected: bool,
) -> Types.BoolArray:
    """Screen a `Graphs` batch in parallel for VF2 subgraph/isomorphism matches against `sub`."""
    n_graphs = len(batch_node_offsets) - 1
    out = np.zeros(n_graphs, dtype=np.bool_)
    for g in prange(n_graphs):  # type: ignore[not-iterable]
        n_s = int(batch_node_offsets[g])
        n_e = int(batch_node_offsets[g + 1])
        e_s = int(batch_edge_offsets[g])
        e_e = int(batch_edge_offsets[g + 1])
        n_edges = e_e - e_s
        local_sources = np.empty(n_edges, dtype=np.uint32)
        local_targets = np.empty(n_edges, dtype=np.uint32)
        for k in range(n_edges):
            local_sources[k] = np.uint32(int(batch_edge_sources[e_s + k]) - n_s)
            local_targets[k] = np.uint32(int(batch_edge_targets[e_s + k]) - n_s)
        found, _ = _vf2_match_kernel(
            sub_nodes,
            sub_sources,
            sub_targets,
            sub_edge_types,
            batch_node_types[n_s:n_e],
            local_sources,
            local_targets,
            batch_edge_types[e_s:e_e],
            node_mask,
            node_wildcard_masks,
            edge_mask,
            edge_wildcard_masks,
            require_bijection,
            undirected,
        )
        out[g] = found
    return out


@Types.njit(parallel=True)
def _wl_smooth_embeddings_kernel(
    features: Types.Float64Array2D,
    node_offsets: Types.UInt32Array,
    edge_sources: Types.UInt32Array,
    edge_targets: Types.UInt32Array,
    edge_offsets: Types.UInt32Array,
    iterations: int,
    smoothing_factor: float,
) -> Types.Float64Array2D:
    """Topologically smooth node features across symmetrized graph neighborhoods in parallel."""
    total_nodes, d = features.shape
    n_graphs = len(node_offsets) - 1
    out_dim = (iterations + 1) * d
    out = np.zeros((total_nodes, out_dim), dtype=np.float64)
    if total_nodes == 0:
        return out

    alpha = smoothing_factor
    one_minus_alpha = 1.0 - smoothing_factor

    for g in prange(n_graphs):  # type: ignore[not-iterable]
        n_s = int(node_offsets[g])
        n_e = int(node_offsets[g + 1])
        n_v = n_e - n_s
        if n_v == 0:
            continue

        for u in range(n_v):
            for col in range(d):
                out[n_s + u, col] = features[n_s + u, col]

        if iterations <= 0:
            continue

        e_s = int(edge_offsets[g])
        e_e = int(edge_offsets[g + 1])
        deg = np.zeros(n_v, dtype=np.float64)
        for k in range(e_s, e_e):
            u = int(edge_sources[k]) - n_s
            v = int(edge_targets[k]) - n_s
            if 0 <= u < n_v and 0 <= v < n_v:
                deg[u] += 1.0
                deg[v] += 1.0

        inv_deg = np.zeros(n_v, dtype=np.float64)
        for u in range(n_v):
            if deg[u] > 0.0:
                inv_deg[u] = 1.0 / deg[u]

        nbr_sum = np.empty((n_v, d), dtype=np.float64)
        for it in range(1, iterations + 1):
            prev_off = (it - 1) * d
            curr_off = it * d
            nbr_sum.fill(0.0)
            for k in range(e_s, e_e):
                u = int(edge_sources[k]) - n_s
                v = int(edge_targets[k]) - n_s
                if 0 <= u < n_v and 0 <= v < n_v:
                    gu = n_s + u
                    gv = n_s + v
                    for col in range(d):
                        nbr_sum[u, col] += out[gv, prev_off + col]
                        nbr_sum[v, col] += out[gu, prev_off + col]

            for u in range(n_v):
                gu = n_s + u
                id_u = inv_deg[u]
                for col in range(d):
                    out[gu, curr_off + col] = one_minus_alpha * out[gu, prev_off + col] + alpha * id_u * nbr_sum[u, col]

    return out


@Types.njit
def _emd_uniform_nd_kernel(
    x: Types.Float64Array2D,
    y: Types.Float64Array2D,
) -> float:
    """Compute exact 1-Wasserstein distance (EMD) between uniform ND point clouds `x` and `y`."""
    n = x.shape[0]
    m = y.shape[0]
    if n == 0 or m == 0:
        return 0.0 if n == m else 1.0
    d = x.shape[1]

    cost = np.empty((n, m), dtype=np.float64)
    for u in range(n):
        for v in range(m):
            sq = 0.0
            for k in range(d):
                diff = x[u, k] - y[v, k]
                sq += diff * diff
            cost[u, v] = np.sqrt(sq)

    if n == 1 and m == 1:
        return float(cost[0, 0])
    if n == 1:
        s = 0.0
        for v in range(m):
            s += cost[0, v]
        return s / float(m)
    if m == 1:
        s = 0.0
        for u in range(n):
            s += cost[u, 0]
        return s / float(n)

    # GCD(n, m) for exact integer transportation supplies and demands
    a_g = n
    b_g = m
    while b_g != 0:
        a_g, b_g = b_g, a_g % b_g
    g = a_g

    s_init = m // g
    d_init = n // g
    tot_flow = n * s_init
    rem_flow = tot_flow

    supply = np.full(n, s_init, dtype=np.int64)
    demand = np.full(m, d_init, dtype=np.int64)
    flow = np.zeros((n, m), dtype=np.int64)

    pi = np.zeros(n, dtype=np.float64)
    sigma = np.empty(m, dtype=np.float64)
    for v in range(m):
        min_c = cost[0, v]
        for u in range(1, n):
            if cost[u, v] < min_c:
                min_c = cost[u, v]
        sigma[v] = min_c

    dist_u = np.empty(n, dtype=np.float64)
    dist_v = np.empty(m, dtype=np.float64)
    vis_u = np.empty(n, dtype=np.bool_)
    vis_v = np.empty(m, dtype=np.bool_)
    prev_u = np.empty(n, dtype=np.int32)
    prev_v = np.empty(m, dtype=np.int32)

    inf = 1e300
    max_augs = 4 * n * m + 16
    aug_step = 0

    while rem_flow > 0 and aug_step < max_augs:
        aug_step += 1
        for u in range(n):
            dist_u[u] = 0.0 if supply[u] > 0 else inf
            vis_u[u] = False
            prev_u[u] = -1
        for v in range(m):
            dist_v[v] = inf
            vis_v[v] = False
            prev_v[v] = -1

        for _ in range(n + m):
            best_dist = inf
            best_is_u = True
            best_idx = -1
            for u in range(n):
                if not vis_u[u] and dist_u[u] < best_dist:
                    best_dist = dist_u[u]
                    best_is_u = True
                    best_idx = u
            for v in range(m):
                if not vis_v[v] and dist_v[v] < best_dist:
                    best_dist = dist_v[v]
                    best_is_u = False
                    best_idx = v

            if best_idx == -1:
                break

            if best_is_u:
                u = best_idx
                vis_u[u] = True
                du = dist_u[u]
                pu = pi[u]
                for v in range(m):
                    if not vis_v[v]:
                        rc = cost[u, v] + pu - sigma[v]
                        if rc < 0.0:
                            rc = 0.0
                        nd = du + rc
                        if nd < dist_v[v]:
                            dist_v[v] = nd
                            prev_v[v] = u
            else:
                v = best_idx
                vis_v[v] = True
                dv = dist_v[v]
                sv = sigma[v]
                for u in range(n):
                    if not vis_u[u] and flow[u, v] > 0:
                        rc = -cost[u, v] + sv - pi[u]
                        if rc < 0.0:
                            rc = 0.0
                        nd = dv + rc
                        if nd < dist_u[u]:
                            dist_u[u] = nd
                            prev_u[u] = v

        best_sink = -1
        best_sink_dist = inf
        for v in range(m):
            if demand[v] > 0 and prev_v[v] != -1 and dist_v[v] < best_sink_dist:
                best_sink_dist = dist_v[v]
                best_sink = v

        if best_sink == -1:
            break

        for u in range(n):
            if vis_u[u] and dist_u[u] < best_sink_dist:
                pi[u] += dist_u[u] - best_sink_dist
        for v in range(m):
            if vis_v[v] and dist_v[v] < best_sink_dist:
                sigma[v] += dist_v[v] - best_sink_dist

        delta = demand[best_sink]
        curr_v = best_sink
        while True:
            curr_u = int(prev_v[curr_v])
            if prev_u[curr_u] == -1:
                if supply[curr_u] < delta:
                    delta = supply[curr_u]
                break
            next_v = int(prev_u[curr_u])
            if flow[curr_u, next_v] < delta:
                delta = flow[curr_u, next_v]
            curr_v = next_v

        demand[best_sink] -= delta
        rem_flow -= int(delta)
        curr_v = best_sink
        while True:
            curr_u = int(prev_v[curr_v])
            flow[curr_u, curr_v] += delta
            if prev_u[curr_u] == -1:
                supply[curr_u] -= delta
                break
            next_v = int(prev_u[curr_u])
            flow[curr_u, next_v] -= delta
            curr_v = next_v

    total_cost = 0.0
    for u in range(n):
        for v in range(m):
            if flow[u, v] > 0:
                total_cost += float(flow[u, v]) * cost[u, v]
    return total_cost / float(tot_flow)


@Types.njit(parallel=True)
def _batch_wwl_distance_kernel(
    embeddings: Types.Float64Array2D,
    node_offsets: Types.UInt32Array,
) -> Types.Float64Array2D:
    """Compute pairwise ND 1-Wasserstein distances across smoothed graph node embeddings in parallel."""
    n_graphs = len(node_offsets) - 1
    dist_matrix = np.zeros((n_graphs, n_graphs), dtype=np.float64)
    if n_graphs == 0:
        return dist_matrix

    for i in prange(n_graphs):  # type: ignore[not-iterable]
        s_i = int(node_offsets[i])
        e_i = int(node_offsets[i + 1])
        x_i = embeddings[s_i:e_i]
        for j in range(i + 1, n_graphs):
            s_j = int(node_offsets[j])
            e_j = int(node_offsets[j + 1])
            d = _emd_uniform_nd_kernel(x_i, embeddings[s_j:e_j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix
