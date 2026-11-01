"""Molecular graph primitives (`Atom`, `Bond`, `BondEdge`, `Mol`, `Mols`)."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, overload

import numpy as np
from numba import prange

from seroverse.core.collections import Spec, SpecEnum, Specs
from seroverse.core.graph.models import TAG_EDGE, BitField, DirectedEdge, Graph, Graphs, IntNode
from seroverse.core.mol.elements import Element
from seroverse.core.types import Types


# Enums ----------------------------------------------------------------------------------------------------------------
class AtomChirality(SpecEnum):
    r"""Tetrahedral stereochemical parity of an [`Atom`][seroverse.core.mol.mol.Atom]."""

    NONE = 0
    CCW = 1  # @ (counter-clockwise)
    CW = 2  # @@ (clockwise)

    @classmethod
    def _compile_specs(cls) -> Specs[Spec]:
        return Specs.from_list([Spec(b""), Spec(b"@"), Spec(b"@@")])


class Hybridization(SpecEnum):
    r"""Orbital hybridization state of an [`Atom`][seroverse.core.mol.mol.Atom]."""

    UNSPECIFIED = 0
    S = 1
    SP = 2
    SP2 = 3
    SP3 = 4
    SP3D = 5
    SP3D2 = 6

    @classmethod
    def _compile_specs(cls) -> Specs[Spec]:
        return Specs.from_list(
            [
                Spec(b"?"),
                Spec(b"s"),
                Spec(b"sp"),
                Spec(b"sp2"),
                Spec(b"sp3"),
                Spec(b"sp3d"),
                Spec(b"sp3d2"),
            ]
        )


class BondOrder(SpecEnum):
    r"""Covalent or non-covalent bond order for a [`Bond`][seroverse.core.mol.mol.Bond]."""

    UNKNOWN = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4
    DATIVE = 5
    HYDROGEN = 6

    @classmethod
    def _compile_specs(cls) -> Specs[Spec]:
        return Specs.from_list(
            [
                Spec(b"~"),
                Spec(b"-"),
                Spec(b"="),
                Spec(b"#"),
                Spec(b":"),
                Spec(b"->"),
                Spec(b"."),
            ]
        )


class BondStereo(SpecEnum):
    r"""Directional cis/trans ($E$/$Z$) stereochemistry for a [`Bond`][seroverse.core.mol.mol.Bond]."""

    NONE = 0
    UP = 1  # /
    DOWN = 2  # \
    ANY = 3  # ~

    @classmethod
    def _compile_specs(cls) -> Specs[Spec]:
        return Specs.from_list([Spec(b""), Spec(b"/"), Spec(b"\\"), Spec(b"~")])


# Bit-Packed Primitives ------------------------------------------------------------------------------------------------
class Atom(IntNode):
    r"""Unified 32-bit integer encoding for an atomic node in a molecular graph.

    Bit Layout:
        - Bits 0-6:   `ELEMENT` (`0..118`, `Element` atomic number or `0 = *`)
        - Bits 7-10:  `CHARGE` (`1..15`, formal charge biased by `+8`, so `8` = neutral `0`, range `-7..+7`)
        - Bits 11-14: `HYDROGENS` (`0..15`, total attached implicit + bracketed hydrogens)
        - Bits 15-16: `CHIRALITY` (`AtomChirality`: `0=NONE`, `1=CCW/@`, `2=CW/@@`)
        - Bits 17-19: `HYBRIDIZATION` (`Hybridization`: `0=UNSPECIFIED..6=SP3D2`)
        - Bit 20:     `AROMATIC` (`1` = aromatic atom, `0` = aliphatic)
        - Bit 21:     `IN_RING` (`1` = member of a topological ring/cycle, `0` = acyclic)
        - Bits 22-30: `ISOTOPE` (`0` = natural abundance, or explicit mass number `1..511`)
        - Bit 31:     `TAG_EDGE` (`0` for primary atom nodes)
    """

    ELEMENT = BitField(shift=0, width=7)
    CHARGE = BitField(shift=7, width=4)
    HYDROGENS = BitField(shift=11, width=4)
    CHIRALITY = BitField(shift=15, width=2)
    HYBRIDIZATION = BitField(shift=17, width=3)
    AROMATIC = BitField(shift=20, width=1)
    IN_RING = BitField(shift=21, width=1)
    ISOTOPE = BitField(shift=22, width=9)
    TAG_EDGE = BitField(shift=31, width=1)

    CHARGE_BIAS: int = 8
    CHEMICAL_MASK: int = int(
        ELEMENT.mask | CHARGE.mask | HYDROGENS.mask | CHIRALITY.mask | AROMATIC.mask | ISOTOPE.mask
    )

    def __new__(
        cls,
        val_or_element: int | np.integer[Any] | str | bytes | Element = Element.C,
        *,
        charge: int = 0,
        hydrogens: int = 0,
        chirality: AtomChirality | int = AtomChirality.NONE,
        hybridization: Hybridization | int = Hybridization.UNSPECIFIED,
        is_aromatic: bool = False,
        in_ring: bool = False,
        isotope: int = 0,
    ) -> Self:
        if (
            isinstance(val_or_element, (int, np.integer))
            and not isinstance(val_or_element, Element)
            and (int(val_or_element) & cls.CHARGE.mask) != 0
            and charge == 0
            and hydrogens == 0
            and chirality == AtomChirality.NONE
            and hybridization == Hybridization.UNSPECIFIED
            and not is_aromatic
            and not in_ring
            and isotope == 0
        ):
            return super().__new__(cls, int(val_or_element))

        elem = val_or_element if isinstance(val_or_element, Element) else Element.from_name(val_or_element)
        packed = cls.pack(
            element=int(elem),
            charge=charge,
            hydrogens=hydrogens,
            chirality=int(chirality),
            hybridization=int(hybridization),
            is_aromatic=is_aromatic,
            in_ring=in_ring,
            isotope=isotope,
        )
        return super().__new__(cls, packed)

    @classmethod
    def pack(
        cls,
        element: int | Element = Element.C,
        charge: int = 0,
        hydrogens: int = 0,
        chirality: int | AtomChirality = AtomChirality.NONE,
        hybridization: int | Hybridization = Hybridization.UNSPECIFIED,
        is_aromatic: bool | int = False,
        in_ring: bool | int = False,
        isotope: int = 0,
        **_kwargs: int,
    ) -> int:
        r"""Pack atomic properties into a 32-bit unsigned integer."""
        if not (-7 <= charge <= 7):
            raise ValueError(f"Atom formal charge must be in [-7, +7], got {charge}")
        return (
            cls.ELEMENT.pack(int(element))
            | cls.CHARGE.pack(int(charge) + cls.CHARGE_BIAS)
            | cls.HYDROGENS.pack(int(hydrogens))
            | cls.CHIRALITY.pack(int(chirality))
            | cls.HYBRIDIZATION.pack(int(hybridization))
            | cls.AROMATIC.pack(int(is_aromatic))
            | cls.IN_RING.pack(int(in_ring))
            | cls.ISOTOPE.pack(int(isotope))
        )

    @property
    def element(self) -> Element:
        r"""Chemical [`Element`][seroverse.core.mol.elements.Element] of this atom."""
        return Element(self.ELEMENT)

    @property
    def symbol(self) -> bytes:
        r"""Canonical atomic symbol as ASCII `bytes` (e.g. `b"C"`, `b"N"`, `b"Fe"`)."""
        return self.element.name_bytes

    @property
    def symbol_str(self) -> str:
        r"""Canonical atomic symbol as a `str`."""
        return self.element.name_str

    @property
    def charge(self) -> int:
        r"""Formal ionic charge in `[-7, +7]`."""
        raw = int(self.CHARGE)
        return (raw - self.CHARGE_BIAS) if raw != 0 else 0

    @property
    def hydrogens(self) -> int:
        r"""Number of attached hydrogen atoms (`0..15`)."""
        return int(self.HYDROGENS)

    @property
    def chirality(self) -> AtomChirality:
        r"""Tetrahedral stereochemical parity."""
        return AtomChirality(self.CHIRALITY)

    @property
    def hybridization(self) -> Hybridization:
        r"""Orbital hybridization state."""
        return Hybridization(self.HYBRIDIZATION)

    @property
    def is_aromatic(self) -> bool:
        r"""Whether this atom is part of a delocalized aromatic pi-system."""
        return bool(self.AROMATIC)

    @property
    def in_ring(self) -> bool:
        r"""Whether this atom lies in a topological ring/cycle."""
        return bool(self.IN_RING)

    @property
    def isotope(self) -> int:
        r"""Explicit mass number (`1..511`), or `0` for natural isotopic abundance."""
        return int(self.ISOTOPE)

    @property
    def atomic_mass(self) -> float:
        r"""Mass of this heavy atom in Daltons (excluding attached implicit hydrogens)."""
        iso = self.isotope
        return float(iso) if iso > 0 else float(self.element.spec.mass)

    @property
    def mass(self) -> float:
        r"""Total mass of this atom plus its attached implicit hydrogens in Daltons."""
        return self.atomic_mass + self.hydrogens * float(Element.H.spec.mass)

    def with_hydrogens(self, hydrogens: int) -> Atom:
        r"""Return a copy of this `Atom` with updated attached `hydrogens` count."""
        cleared = int(self) & ~type(self).HYDROGENS.mask
        return Atom(cleared | type(self).HYDROGENS.pack(int(hydrogens)))

    def with_charge(self, charge: int) -> Atom:
        r"""Return a copy of this `Atom` with updated formal `charge`."""
        if not (-7 <= charge <= 7):
            raise ValueError(f"Atom formal charge must be in [-7, +7], got {charge}")
        cleared = int(self) & ~type(self).CHARGE.mask
        return Atom(cleared | type(self).CHARGE.pack(int(charge) + self.CHARGE_BIAS))

    def with_in_ring(self, in_ring: bool = True) -> Atom:
        r"""Return a copy of this `Atom` with the `IN_RING` bit set or cleared."""
        mask = type(self).IN_RING.mask
        return Atom((int(self) | mask) if in_ring else (int(self) & ~mask))

    def with_aromatic(self, is_aromatic: bool = True) -> Atom:
        r"""Return a copy of this `Atom` with the `AROMATIC` bit set or cleared."""
        mask = type(self).AROMATIC.mask
        return Atom((int(self) | mask) if is_aromatic else (int(self) & ~mask))

    def with_hybridization(self, hybridization: Hybridization | int) -> Atom:
        r"""Return a copy of this `Atom` with updated orbital `hybridization`."""
        cleared = int(self) & ~type(self).HYBRIDIZATION.mask
        return Atom(cleared | type(self).HYBRIDIZATION.pack(int(hybridization)))

    def with_chirality(self, chirality: AtomChirality | int) -> Atom:
        r"""Return a copy of this `Atom` with updated tetrahedral `chirality`."""
        cleared = int(self) & ~type(self).CHIRALITY.mask
        return Atom(cleared | type(self).CHIRALITY.pack(int(chirality)))

    def __repr__(self) -> str:
        sym = self.symbol_str.lower() if self.is_aromatic else self.symbol_str
        iso = str(self.isotope) if self.isotope > 0 else ""
        chg = f"{self.charge:+d}" if self.charge != 0 else ""
        h_str = f"H{self.hydrogens}" if self.hydrogens > 1 else ("H" if self.hydrogens == 1 else "")
        return f"Atom({iso}{sym}{h_str}{chg})"


class Bond(IntNode):
    r"""Unified 32-bit integer encoding for a chemical bond (`TAG_EDGE` set).

    Bit Layout:
        - Bits 0-2: `ORDER` (`BondOrder`: `0=UNKNOWN`, `1=SINGLE`, `2=DOUBLE`, `3=TRIPLE`, `4=AROMATIC`, `5=DATIVE`, `6=HYDROGEN`)
        - Bits 3-4: `STEREO` (`BondStereo`: `0=NONE`, `1=UP (/)`, `2=DOWN (\)`, `3=ANY (~)`)
        - Bit 5:    `CONJUGATED` (`1` = conjugated pi-bond, `0` = non-conjugated)
        - Bit 6:    `IN_RING` (`1` = ring bond, `0` = acyclic bond)
        - Bit 31:   `TAG_EDGE` (`1` = edge node discriminator in bipartite graphs)
    """

    ORDER = BitField(shift=0, width=3)
    STEREO = BitField(shift=3, width=2)
    CONJUGATED = BitField(shift=5, width=1)
    IN_RING = BitField(shift=6, width=1)
    TAG_EDGE = BitField(shift=31, width=1)

    CHEMICAL_MASK: int = int(ORDER.mask | STEREO.mask | TAG_EDGE.mask)

    def __new__(
        cls,
        val_or_order: int | np.integer[Any] | str | bytes | BondOrder = BondOrder.SINGLE,
        *,
        stereo: BondStereo | int = BondStereo.NONE,
        is_conjugated: bool = False,
        in_ring: bool = False,
    ) -> Self:
        if (
            isinstance(val_or_order, (int, np.integer))
            and not isinstance(val_or_order, BondOrder)
            and (int(val_or_order) & TAG_EDGE) != 0
            and stereo == BondStereo.NONE
            and not is_conjugated
            and not in_ring
        ):
            return super().__new__(cls, int(val_or_order))

        order = val_or_order if isinstance(val_or_order, BondOrder) else BondOrder.from_name(val_or_order)
        packed = cls.pack(
            order=int(order),
            stereo=int(stereo),
            is_conjugated=is_conjugated,
            in_ring=in_ring,
        )
        return super().__new__(cls, packed)

    @classmethod
    def pack(
        cls,
        order: int | BondOrder = BondOrder.SINGLE,
        stereo: int | BondStereo = BondStereo.NONE,
        is_conjugated: bool | int = False,
        in_ring: bool | int = False,
        **_kwargs: int,
    ) -> int:
        r"""Pack bond properties into a 32-bit unsigned integer with `TAG_EDGE` set."""
        return (
            cls.ORDER.pack(int(order))
            | cls.STEREO.pack(int(stereo))
            | cls.CONJUGATED.pack(int(is_conjugated))
            | cls.IN_RING.pack(int(in_ring))
            | cls.TAG_EDGE.pack(1)
        )

    @property
    def order(self) -> BondOrder:
        r"""Covalent or non-covalent [`BondOrder`][seroverse.core.mol.mol.BondOrder]."""
        return BondOrder(self.ORDER)

    @property
    def stereo(self) -> BondStereo:
        r"""Cis/trans directional [`BondStereo`][seroverse.core.mol.mol.BondStereo]."""
        return BondStereo(self.STEREO)

    @property
    def is_conjugated(self) -> bool:
        r"""Whether this bond participates in a conjugated system."""
        return bool(self.CONJUGATED)

    @property
    def in_ring(self) -> bool:
        r"""Whether this bond lies within a topological ring."""
        return bool(self.IN_RING)

    @property
    def is_aromatic(self) -> bool:
        r"""Whether this bond has `BondOrder.AROMATIC`."""
        return self.order == BondOrder.AROMATIC

    @property
    def valence_contribution(self) -> float:
        r"""Effective valence electron contribution of this bond (`1.0`, `2.0`, `3.0`, or `1.5` for aromatic)."""
        ord_val = int(self.ORDER)
        if ord_val == int(BondOrder.AROMATIC):
            return 1.5
        if 1 <= ord_val <= 3:
            return float(ord_val)
        return 0.0

    def with_in_ring(self, in_ring: bool = True) -> Bond:
        r"""Return a copy of this `Bond` with the `IN_RING` bit set or cleared."""
        mask = type(self).IN_RING.mask
        return Bond((int(self) | mask) if in_ring else (int(self) & ~mask))

    def with_stereo(self, stereo: BondStereo | int) -> Bond:
        r"""Return a copy of this `Bond` with updated `stereo`."""
        cleared = int(self) & ~type(self).STEREO.mask
        return Bond(cleared | type(self).STEREO.pack(int(stereo)))

    def with_conjugated(self, is_conjugated: bool = True) -> Bond:
        r"""Return a copy of this `Bond` with the `CONJUGATED` bit set or cleared."""
        mask = type(self).CONJUGATED.mask
        return Bond((int(self) | mask) if is_conjugated else (int(self) & ~mask))

    def __repr__(self) -> str:
        return f"Bond({self.order.name_str})"


@dataclass(slots=True, frozen=True)
class BondEdge(DirectedEdge):
    r"""Directed chemical bond edge connecting `source` atom index to `target` atom index."""

    source: int
    target: int
    edge_type: Bond = field(default_factory=Bond)


# Molecular Graph & SoA Container --------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Mol(Graph):
    r"""Base chemical graph view representing a molecule or residue-level biopolymer (e.g. `Glycan`).

    Inherits topological CSR, spectral distance, and VF2 substructure/isomorphism primitives
    from [`Graph`][seroverse.core.graph.models.Graph] and adds:
    - Canonical linear string identity (`canonical: bytes`).
    - Physicochemical properties (`mass`, `charge`, `formula`, `composition_mask`).
    - RDKit-style substructure matching (`has_substruct_match`, `get_substruct_match`, `is_isomorphic`).
    - Continuous Wasserstein Weisfeiler-Lehman (`wwl_distance`) chemical similarity.
    - OpenSMILES parsing and canonical SMILES serialization (`from_smiles`, `to_smiles`).
    """

    _node_cls: ClassVar[type[Any]] = Atom
    _edge_cls: ClassVar[type[Any]] = BondEdge
    _edge_type_cls: ClassVar[type[Any]] = Bond
    _spec_enum: ClassVar[type[SpecEnum]] = Element
    _type_mask: ClassVar[int] = int(Atom.ELEMENT.mask)
    _undirected_matching: ClassVar[bool] = True

    nodes: tuple[Any, ...] = ()
    edges: tuple[Any, ...] = ()
    canonical: bytes = b""

    @property
    def num_atoms(self) -> int:
        r"""Number of nodes (atoms or residues) in this molecule."""
        return len(self.nodes)

    @property
    def num_bonds(self) -> int:
        r"""Number of edges (bonds or linkages) in this molecule."""
        return len(self.edges)

    @property
    def composition_mask(self) -> int:
        r"""Bitmask of all monomer/element ontology indices (`int(node) & _type_mask`) present in `self.nodes`."""
        mask = 0
        t_mask = self._type_mask
        for n in self.nodes:
            mask |= 1 << (int(n) & t_mask)
        return mask

    @property
    def mass(self) -> float:
        r"""Total molecular weight of `self` in Daltons (Da)."""
        if not self.nodes:
            return 0.0
        if self._node_cls is Atom:
            return float(sum(n.mass for n in self.nodes))
        masses = getattr(self._spec_enum.specs(), "mass", getattr(self._spec_enum.specs(), "masses", None))
        if masses is None:
            return 0.0
        t_mask = self._type_mask
        return float(sum(float(masses[int(n) & t_mask]) for n in self.nodes))

    @property
    def charge(self) -> int:
        r"""Total formal ionic charge of `self`."""
        if not self.nodes:
            return 0
        if self._node_cls is Atom:
            return sum(n.charge for n in self.nodes)
        charges = getattr(self._spec_enum.specs(), "charge", getattr(self._spec_enum.specs(), "charges", None))
        if charges is None:
            return 0
        t_mask = self._type_mask
        return sum(int(charges[int(n) & t_mask]) for n in self.nodes)

    @property
    def formula(self) -> str:
        r"""Hill-system molecular formula string (e.g. `'C2H6O'`, `'C6H12O6'`, `'NH4+'`)."""
        if not self.nodes:
            return ""
        counts: dict[str, int] = {}
        total_h = 0
        for n in self.nodes:
            if isinstance(n, Atom):
                sym = n.symbol_str
                if sym == "H":
                    total_h += 1 + n.hydrogens
                elif sym != "*":
                    counts[sym] = counts.get(sym, 0) + 1
                    total_h += n.hydrogens
            else:
                sym = self._spec_enum(int(n) & self._type_mask).name_str
                counts[sym] = counts.get(sym, 0) + 1
        if total_h > 0:
            counts["H"] = counts.get("H", 0) + total_h

        parts: list[str] = []
        if "C" in counts:
            c_cnt = counts.pop("C")
            parts.append(f"C{c_cnt}" if c_cnt > 1 else "C")
            if "H" in counts:
                h_cnt = counts.pop("H")
                parts.append(f"H{h_cnt}" if h_cnt > 1 else "H")
        for sym in sorted(counts):
            cnt = counts[sym]
            parts.append(f"{sym}{cnt}" if cnt > 1 else sym)

        net_charge = self.charge
        if net_charge > 0:
            parts.append("+" if net_charge == 1 else f"+{net_charge}")
        elif net_charge < 0:
            parts.append("-" if net_charge == -1 else f"{net_charge}")
        return "".join(parts)

    def ring_atoms_and_bonds(self) -> tuple[frozenset[int], frozenset[int]]:
        r"""Identify 0-based atom indices and bond indices belonging to any topological ring ($O(V + E)$)."""
        n = len(self.nodes)
        m = len(self.edges)
        if n == 0 or m == 0:
            return frozenset(), frozenset()

        adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for k, e in enumerate(self.edges):
            if 0 <= e.source < n and 0 <= e.target < n and e.source != e.target:
                adj[e.source].append((e.target, k))
                adj[e.target].append((e.source, k))

        depth: dict[int, int] = {}
        parent_node: dict[int, int] = {}
        parent_edge: dict[int, int] = {}
        ring_atoms: set[int] = set()
        ring_bonds: set[int] = set()

        for start in range(n):
            if start in depth:
                continue
            depth[start] = 0
            stack: list[tuple[int, int]] = [(start, 0)]
            while stack:
                u, idx = stack[-1]
                nbrs = adj[u]
                if idx < len(nbrs):
                    stack[-1] = (u, idx + 1)
                    v, edge_k = nbrs[idx]
                    if edge_k == parent_edge.get(u, -1):
                        continue
                    if v not in depth:
                        depth[v] = depth[u] + 1
                        parent_node[v] = u
                        parent_edge[v] = edge_k
                        stack.append((v, 0))
                    elif depth[v] < depth[u]:
                        ring_bonds.add(edge_k)
                        ring_atoms.add(u)
                        ring_atoms.add(v)
                        curr = u
                        while curr != v and curr in parent_node:
                            ring_atoms.add(curr)
                            ring_bonds.add(parent_edge[curr])
                            curr = parent_node[curr]
                        ring_atoms.add(v)
                else:
                    stack.pop()

        return frozenset(ring_atoms), frozenset(ring_bonds)

    @classmethod
    def _substruct_masks(
        cls,
        *,
        allow_wildcards: bool = True,
    ) -> tuple[int, tuple[int, ...], int, tuple[int, ...]]:
        r"""Return `(node_mask, node_wildcard_masks, edge_mask, edge_wildcard_masks)` for VF2 matching."""
        if cls._node_cls is Atom:
            if allow_wildcards:
                return (
                    int(Atom.AROMATIC.mask),
                    (int(Atom.ELEMENT.mask), int(Atom.CHIRALITY.mask), int(Atom.ISOTOPE.mask)),
                    0,
                    (int(Bond.ORDER.mask), int(Bond.STEREO.mask)),
                )
            return (
                int(Atom.CHEMICAL_MASK),
                (),
                int(Bond.CHEMICAL_MASK),
                (),
            )
        return (0xFFFFFFFF, (), 0xFFFFFFFF, ())

    @classmethod
    def _extract_node_features(cls, node_types: Types.UInt32Array) -> Types.Float32Array2D:
        r"""Extract the continuous `(N_nodes, D)` `float32` node feature matrix for WWL smoothing."""
        specs: Any = cls._spec_enum.specs()
        base_feats: Types.Float32Array2D = specs.feature_matrix[node_types & cls._type_mask]
        if cls._node_cls is not Atom:
            return base_feats
        n = len(node_types)
        out = np.empty((n, base_feats.shape[1] + 5), dtype=np.float32)
        out[:, : base_feats.shape[1]] = base_feats
        raw_chg = Atom.CHARGE.extract_array(node_types).astype(np.float32)
        out[:, base_feats.shape[1]] = np.where(raw_chg > 0, (raw_chg - 8.0) / 4.0, 0.0)
        out[:, base_feats.shape[1] + 1] = Atom.HYDROGENS.extract_array(node_types) / np.float32(4.0)
        out[:, base_feats.shape[1] + 2] = Atom.AROMATIC.extract_array(node_types).astype(np.float32)
        out[:, base_feats.shape[1] + 3] = Atom.IN_RING.extract_array(node_types).astype(np.float32)
        out[:, base_feats.shape[1] + 4] = Atom.HYBRIDIZATION.extract_array(node_types) / np.float32(6.0)
        return out

    def get_substruct_match(
        self,
        query: Self,
        *,
        allow_wildcards: bool = True,
    ) -> dict[int, int] | None:
        r"""Return an injective substructure atom mapping `{query_idx: self_idx}`, or `None` if not found."""
        # Fast O(1) composition bitmask screen (ignoring bit 0 wildcard)
        q_mask = query.composition_mask & ~1
        if (q_mask & ~self.composition_mask) != 0:
            return None
        n_mask, nw_masks, e_mask, ew_masks = self._substruct_masks(allow_wildcards=allow_wildcards)
        return query.find_subgraph_mapping(
            self,
            node_mask=n_mask,
            node_wildcard_masks=nw_masks,
            edge_mask=e_mask,
            edge_wildcard_masks=ew_masks,
            require_bijection=False,
            undirected=self._undirected_matching,
        )

    def has_substruct_match(
        self,
        query: Self,
        *,
        allow_wildcards: bool = True,
    ) -> bool:
        r"""Return True if `query` is a chemical substructure of `self`."""
        q_mask = query.composition_mask & ~1
        if (q_mask & ~self.composition_mask) != 0:
            return False
        n_mask, nw_masks, e_mask, ew_masks = self._substruct_masks(allow_wildcards=allow_wildcards)
        return query.match_subgraph(
            self,
            node_mask=n_mask,
            node_wildcard_masks=nw_masks,
            edge_mask=e_mask,
            edge_wildcard_masks=ew_masks,
            require_bijection=False,
            undirected=self._undirected_matching,
        )

    def is_isomorphic(
        self,
        other: Self,
        *,
        allow_wildcards: bool = False,
    ) -> bool:
        r"""Return True if `self` and `other` are chemically isomorphic graphs."""
        if len(self.nodes) != len(other.nodes) or len(self.edges) != len(other.edges):
            return False
        if not allow_wildcards and self.composition_mask != other.composition_mask:
            return False
        n_mask, nw_masks, e_mask, ew_masks = self._substruct_masks(allow_wildcards=allow_wildcards)
        return self.match_subgraph(
            other,
            node_mask=n_mask,
            node_wildcard_masks=nw_masks,
            edge_mask=e_mask,
            edge_wildcard_masks=ew_masks,
            require_bijection=True,
            undirected=self._undirected_matching,
        )

    def wwl_distance(self, other: Self, iterations: int = 2) -> float:
        r"""Compute the continuous Wasserstein Weisfeiler-Lehman (WWL) distance between `self` and `other`."""
        if len(self.nodes) == 0 and len(other.nodes) == 0:
            return 0.0
        if len(self.nodes) == 0 or len(other.nodes) == 0:
            return 1.0
        batch = Mols.from_list([self, other])
        feats = self._extract_node_features(batch.node_types)
        return float(batch.wwl_distance_matrix(feats, iterations=iterations)[0, 1])

    @classmethod
    def from_smiles(cls, smiles: str | bytes) -> Self:
        r"""Parse an OpenSMILES string or byte-string into a [`Mol`][seroverse.core.mol.mol.Mol] instance."""
        from seroverse.core.io.smiles import parse_smiles

        nodes, edges, canonical = parse_smiles(smiles)
        return cls(nodes=nodes, edges=edges, canonical=canonical)

    def to_smiles(self) -> bytes:
        r"""Serialize this [`Mol`][seroverse.core.mol.mol.Mol] into a deterministic canonical OpenSMILES byte-string."""
        from seroverse.core.io.smiles import format_smiles

        return format_smiles(self.nodes, self.edges)

    def __repr__(self) -> str:
        if self.canonical:
            return f"{self.__class__.__name__}({self.canonical.decode('ascii', errors='replace')!r})"
        return f"{self.__class__.__name__}({len(self.nodes)} atoms, {len(self.edges)} bonds)"


@dataclass(slots=True, frozen=True)
class Mols[T: Mol = Mol](Graphs[T]):
    r"""Structure-of-Arrays (SoA) batch container for [`Mol`][seroverse.core.mol.mol.Mol] (and `Glycan`) datasets.

    Extends [`Graphs`][seroverse.core.graph.models.Graphs] with:
    - Contiguous `canonicals: BytesArray` storage and lazy $O(1)$ lookup by canonical byte-string/str.
    - Vectorized batch `masses` and `charges` properties.
    - Parallel VF2 substructure screening (`has_substruct_match`).
    - Zero-SciPy parallel Wasserstein Weisfeiler-Lehman distance matrix (`wwl_distance`).
    """

    _scalar_type: ClassVar[type[Any] | None] = Mol

    node_types: Types.UInt32Array
    node_offsets: Types.UInt32Array
    edge_sources: Types.UInt32Array
    edge_targets: Types.UInt32Array
    edge_types: Types.UInt32Array
    edge_offsets: Types.UInt32Array
    canonicals: Types.BytesArray
    _lookup_cache: dict[bytes, int] | None = field(default=None, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.canonicals, np.ndarray):
            arr = np.array(self.canonicals, dtype="S") if len(self.canonicals) > 0 else np.empty(0, dtype="S1")
            object.__setattr__(self, "canonicals", arr)

    def _get_lookup(self) -> dict[bytes, int]:
        r"""Build and cache the `canonical -> index` lookup dictionary."""
        cached = self._lookup_cache
        if cached is not None:
            return cached
        lookup: dict[bytes, int] = {}
        for i in range(len(self)):
            key = bytes(self.canonicals[i]).strip()
            if key and key not in lookup:
                lookup[key] = i
        object.__setattr__(self, "_lookup_cache", lookup)
        return lookup

    def index(self, key: str | bytes | int | np.integer[Any]) -> int:
        r"""Resolve a canonical string/bytes key or integer index to its 0-based position in the batch."""
        if isinstance(key, (int, np.integer)):
            return self._resolve_index(key)
        b = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        norm = b.strip()
        lookup = self._get_lookup()
        if norm in lookup:
            return lookup[norm]
        raise KeyError(f"Unknown {type(self).__name__} canonical key: {key!r}")

    def resolve_index(self, key: str | bytes | int | np.integer[Any], default: int = -1) -> int:
        r"""Resolve a canonical key or integer index, returning `default` if not found."""
        try:
            return self.index(key)
        except KeyError, IndexError:
            return default

    def __contains__(self, key: object) -> bool:
        r"""Return True if `key` (`str`, `bytes`, or `Mol`) exists in this batch."""
        if isinstance(key, Mol):
            return key.canonical.strip() in self._get_lookup()
        if isinstance(key, (str, bytes)):
            b = key.encode("utf-8") if isinstance(key, str) else bytes(key)
            return b.strip() in self._get_lookup()
        return False

    @overload
    def __getitem__(self, item: int | np.integer[Any] | str | bytes) -> T: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self,
        item: int | np.integer[Any] | str | bytes | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> T | Self: ...

    def __getitem__(
        self,
        item: int | np.integer[Any] | str | bytes | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> T | Self:
        r"""Access a single `Mol` view `T` by integer index or canonical string/bytes, or slice `Self`."""
        if isinstance(item, (int, np.integer)):
            return self._get_scalar(self._resolve_index(item))
        if isinstance(item, (str, bytes)):
            return self._get_scalar(self.index(item))
        return self._get_slice(item)

    @property
    def masses(self) -> Types.Float32Array:
        r"""Vectorized molecular weight (in Da) for each molecule in the batch."""
        scalar_cls = self._scalar_type or Mol
        if issubclass(scalar_cls, Mol) and scalar_cls._node_cls is Atom:
            return _batch_atom_masses_kernel(
                self.node_types,
                self.node_offsets,
                Element.specs().masses,
                np.float32(Element.H.spec.mass),
            )
        specs = scalar_cls._spec_enum.specs()
        spec_masses = getattr(specs, "mass", getattr(specs, "masses", None))
        if spec_masses is None:
            return np.zeros(len(self), dtype=np.float32)
        return _batch_spec_masses_kernel(
            self.node_types,
            self.node_offsets,
            np.ascontiguousarray(spec_masses, dtype=np.float32),
            np.uint32(scalar_cls._type_mask),
        )

    @property
    def charges(self) -> Types.Int32Array:
        r"""Vectorized total formal charge for each molecule in the batch."""
        scalar_cls = self._scalar_type or Mol
        if issubclass(scalar_cls, Mol) and scalar_cls._node_cls is Atom:
            return _batch_atom_charges_kernel(self.node_types, self.node_offsets)
        specs = scalar_cls._spec_enum.specs()
        spec_charges = getattr(specs, "charge", getattr(specs, "charges", None))
        if spec_charges is None:
            return np.zeros(len(self), dtype=np.int32)
        return _batch_spec_charges_kernel(
            self.node_types,
            self.node_offsets,
            np.ascontiguousarray(spec_charges, dtype=np.int8),
            np.uint32(scalar_cls._type_mask),
        )

    def node_features(self) -> Types.Float32Array2D:
        r"""Extract the continuous `(total_nodes, D)` `float32` node feature matrix for the batch."""
        scalar_cls = self._scalar_type or Mol
        return scalar_cls._extract_node_features(self.node_types)

    def wwl_distance(
        self,
        iterations: int = 2,
        smoothing_factor: float = 0.5,
    ) -> Types.Float64Array2D:
        r"""Compute the all-vs-all Wasserstein Weisfeiler-Lehman (WWL) distance matrix across the batch."""
        if len(self.node_types) == 0:
            return np.zeros((len(self), len(self)), dtype=np.float64)
        return self.wwl_distance_matrix(
            self.node_features(),
            iterations=iterations,
            smoothing_factor=smoothing_factor,
        )

    def has_substruct_match(
        self,
        query: T,
        *,
        allow_wildcards: bool = True,
    ) -> Types.BoolArray:
        r"""Screen all molecules in this batch in parallel for a substructure match with `query`."""
        scalar_cls = self._scalar_type or Mol
        n_mask, nw_masks, e_mask, ew_masks = scalar_cls._substruct_masks(allow_wildcards=allow_wildcards)
        return self.match_subgraph_batch(
            query,
            node_mask=n_mask,
            node_wildcard_masks=nw_masks,
            edge_mask=e_mask,
            edge_wildcard_masks=ew_masks,
            require_bijection=False,
            undirected=scalar_cls._undirected_matching,
        )

    @classmethod
    def from_smiles(cls, smiles_list: Iterable[str | bytes]) -> Self:
        r"""Parse an iterable of OpenSMILES strings/bytes into a contiguous [`Mols`][seroverse.core.mol.mol.Mols] batch."""
        scalar_cls: Any = cls._scalar_type or Mol
        return cls.from_list([scalar_cls.from_smiles(s) for s in smiles_list])


# Kernels --------------------------------------------------------------------------------------------------------------
@Types.njit(parallel=True)
def _batch_atom_masses_kernel(
    node_types: Types.UInt32Array,
    node_offsets: Types.UInt32Array,
    element_masses: Types.Float32Array,
    h_mass: np.float32,
) -> Types.Float32Array:
    r"""Compute molecular weights across a `Mols` batch of bit-packed `Atom` nodes in parallel."""
    n_mols = len(node_offsets) - 1
    out = np.zeros(n_mols, dtype=np.float32)
    for i in prange(n_mols):  # type: ignore[not-iterable]
        s = int(node_offsets[i])
        e = int(node_offsets[i + 1])
        m = np.float32(0.0)
        for k in range(s, e):
            val = int(node_types[k])
            el = val & 0x7F
            h_cnt = (val >> 11) & 0xF
            iso = (val >> 22) & 0x1FF
            at_m = np.float32(iso) if iso > 0 else element_masses[el]
            m += at_m + np.float32(h_cnt) * h_mass
        out[i] = m
    return out


@Types.njit(parallel=True)
def _batch_spec_masses_kernel(
    node_types: Types.UInt32Array,
    node_offsets: Types.UInt32Array,
    spec_masses: Types.Float32Array,
    type_mask: np.uint32,
) -> Types.Float32Array:
    r"""Compute molecular weights across a `Mols` subclass batch using `Specs` masses in parallel."""
    n_mols = len(node_offsets) - 1
    out = np.zeros(n_mols, dtype=np.float32)
    for i in prange(n_mols):  # type: ignore[not-iterable]
        s = int(node_offsets[i])
        e = int(node_offsets[i + 1])
        m = np.float32(0.0)
        for k in range(s, e):
            idx = int(node_types[k] & type_mask)
            m += spec_masses[idx]
        out[i] = m
    return out


@Types.njit(parallel=True)
def _batch_atom_charges_kernel(
    node_types: Types.UInt32Array,
    node_offsets: Types.UInt32Array,
) -> Types.Int32Array:
    r"""Compute total formal charges across a `Mols` batch of bit-packed `Atom` nodes in parallel."""
    n_mols = len(node_offsets) - 1
    out = np.zeros(n_mols, dtype=np.int32)
    for i in prange(n_mols):  # type: ignore[not-iterable]
        s = int(node_offsets[i])
        e = int(node_offsets[i + 1])
        c = 0
        for k in range(s, e):
            raw_c = (int(node_types[k]) >> 7) & 0xF
            if raw_c != 0:
                c += raw_c - 8
        out[i] = c
    return out


@Types.njit(parallel=True)
def _batch_spec_charges_kernel(
    node_types: Types.UInt32Array,
    node_offsets: Types.UInt32Array,
    spec_charges: Types.Int8Array,
    type_mask: np.uint32,
) -> Types.Int32Array:
    r"""Compute total formal charges across a `Mols` subclass batch using `Specs` charges in parallel."""
    n_mols = len(node_offsets) - 1
    out = np.zeros(n_mols, dtype=np.int32)
    for i in prange(n_mols):  # type: ignore[not-iterable]
        s = int(node_offsets[i])
        e = int(node_offsets[i + 1])
        c = 0
        for k in range(s, e):
            idx = int(node_types[k] & type_mask)
            c += int(spec_charges[idx])
        out[i] = c
    return out
