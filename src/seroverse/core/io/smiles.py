"""OpenSMILES parser, canonical SMILES serializer, and streaming `SmilesReader` / `SmilesWriter`."""

from collections.abc import Iterator, Sequence
from typing import ClassVar

from seroverse.core.io.base import BaseReader, BaseWriter
from seroverse.core.mol.elements import Element
from seroverse.core.mol.mol import (
    Atom,
    AtomChirality,
    Bond,
    BondEdge,
    BondOrder,
    BondStereo,
    Hybridization,
    Mol,
    Mols,
)

# OpenSMILES Organic Subset & Valence Tables ---------------------------------------------------------------------------
_ORGANIC_ALIPHATIC: dict[str, Element] = {
    "B": Element.B,
    "C": Element.C,
    "N": Element.N,
    "O": Element.O,
    "P": Element.P,
    "S": Element.S,
    "F": Element.F,
    "Cl": Element.CL,
    "Br": Element.BR,
    "I": Element.I,
    "*": Element.WILDCARD,
}

_ORGANIC_AROMATIC: dict[str, Element] = {
    "b": Element.B,
    "c": Element.C,
    "n": Element.N,
    "o": Element.O,
    "p": Element.P,
    "s": Element.S,
}

_DEFAULT_VALENCES: dict[Element, tuple[int, ...]] = {
    Element.B: (3,),
    Element.C: (4,),
    Element.N: (3, 5),
    Element.O: (2,),
    Element.P: (3, 5),
    Element.S: (2, 4, 6),
    Element.F: (1,),
    Element.CL: (1,),
    Element.BR: (1,),
    Element.I: (1,),
}

_BOND_CHARS: dict[str, tuple[BondOrder, BondStereo]] = {
    "-": (BondOrder.SINGLE, BondStereo.NONE),
    "=": (BondOrder.DOUBLE, BondStereo.NONE),
    "#": (BondOrder.TRIPLE, BondStereo.NONE),
    ":": (BondOrder.AROMATIC, BondStereo.NONE),
    "~": (BondOrder.UNKNOWN, BondStereo.ANY),
    "/": (BondOrder.SINGLE, BondStereo.UP),
    "\\": (BondOrder.SINGLE, BondStereo.DOWN),
}


def _implicit_hydrogens(elem: Element, is_aromatic: bool, bond_valence: float, n_aro_bonds: int) -> int:
    r"""Calculate standard OpenSMILES implicit hydrogen count for an unbracketed organic-subset atom."""
    if elem == Element.WILDCARD:
        return 0
    if is_aromatic:
        if elem in (Element.C, Element.B):
            target = 4.0 if elem == Element.C else 3.0
            rem = target - bond_valence
            return max(0, int(round(rem))) if n_aro_bonds >= 2 else max(0, int(target - int(bond_valence) - 1))
        # Unbracketed aromatic n, o, p, s have 0 implicit hydrogens in OpenSMILES (e.g. pyridine vs [nH] pyrrole)
        return 0
    valences = _DEFAULT_VALENCES.get(elem)
    if not valences:
        return 0
    bv = int(round(bond_valence))
    for v in valences:
        if v >= bv:
            return v - bv
    return 0


def _parse_bracket_atom(token: str) -> tuple[Element, int, AtomChirality, int, int, bool]:
    r"""Parse `[isotope? symbol chirality? H_count? charge? :class?]` into atomic properties."""
    n = len(token)
    pos = 0

    # 1. Optional isotope digits
    iso_start = pos
    while pos < n and token[pos].isdigit():
        pos += 1
    isotope = int(token[iso_start:pos]) if pos > iso_start else 0

    # 2. Element symbol
    if pos >= n:
        raise ValueError(f"Empty or invalid bracketed atom: [{token}]")
    if token[pos] == "*":
        elem = Element.WILDCARD
        is_aromatic = False
        pos += 1
    elif token[pos].islower():
        is_aromatic = True
        if pos + 2 <= n and token[pos : pos + 2] in ("se", "as"):
            sym = token[pos : pos + 2]
            pos += 2
        else:
            sym = token[pos]
            pos += 1
        elem = Element.from_name(sym)
    elif token[pos].isupper():
        is_aromatic = False
        if pos + 2 <= n and token[pos + 1].islower() and token[pos : pos + 2] not in ("Ch",):
            # Check if 2-letter symbol is a valid element (note: 'H' followed by nothing is 1-letter, 'He'/'Hf'/etc. are 2-letter)
            cand2 = token[pos : pos + 2]
            try:
                elem = Element.from_name(cand2)
                pos += 2
            except KeyError:
                elem = Element.from_name(token[pos])
                pos += 1
        else:
            elem = Element.from_name(token[pos])
            pos += 1
    else:
        raise ValueError(f"Unexpected character in bracketed atom [{token}] at position {pos}")

    # 3. Optional chirality (@ or @@)
    chirality = AtomChirality.NONE
    if pos < n and token[pos] == "@":
        if pos + 1 < n and token[pos + 1] == "@":
            chirality = AtomChirality.CW
            pos += 2
        else:
            chirality = AtomChirality.CCW
            pos += 1

    # 4. Optional explicit hydrogen count (H or H<digit>)
    hydrogens = 0
    if pos < n and token[pos] == "H":
        pos += 1
        h_start = pos
        while pos < n and token[pos].isdigit():
            pos += 1
        hydrogens = int(token[h_start:pos]) if pos > h_start else 1

    # 5. Optional formal charge (+, ++, +2, -, --, -2)
    charge = 0
    if pos < n and token[pos] in ("+", "-"):
        sign = 1 if token[pos] == "+" else -1
        sign_char = token[pos]
        pos += 1
        if pos < n and token[pos].isdigit():
            c_start = pos
            while pos < n and token[pos].isdigit():
                pos += 1
            charge = sign * int(token[c_start:pos])
        else:
            cnt = 1
            while pos < n and token[pos] == sign_char:
                cnt += 1
                pos += 1
            charge = sign * cnt

    # 6. Optional atom class (:1)
    if pos < n and token[pos] == ":":
        pos += 1
        while pos < n and token[pos].isdigit():
            pos += 1

    if pos != n:
        raise ValueError(f"Unparsed trailing characters in bracketed atom [{token}]: {token[pos:]!r}")

    return elem, isotope, chirality, hydrogens, charge, is_aromatic


def parse_smiles(smiles: str | bytes | bytearray) -> tuple[tuple[Atom, ...], tuple[BondEdge, ...], bytes]:
    r"""Parse an OpenSMILES string or byte-string into `(nodes, edges, canonical_smiles_bytes)`."""
    raw = smiles.decode("ascii") if isinstance(smiles, (bytes, bytearray)) else str(smiles)
    s = raw.strip()
    if not s:
        return (), (), b""
    # Extract first whitespace-delimited token (ignoring trailing molecule ID/name)
    s = s.split()[0]

    # Mutable atom records: [elem, isotope, chirality, hydrogens, charge, is_aromatic, is_bracket]
    atom_specs: list[list[int | bool | Element | AtomChirality]] = []
    # Raw edges: [u, v, order, stereo]
    raw_edges: list[list[int | BondOrder | BondStereo]] = []

    branch_stack: list[int] = []
    ring_openings: dict[int, tuple[int, BondOrder | None, BondStereo]] = {}
    curr_atom: int = -1
    pending_bond: tuple[BondOrder | None, BondStereo] | None = None
    disconnected = False

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]

        if ch == "(":
            if curr_atom < 0:
                raise ValueError(f"Branch '(' before any atom in SMILES: {s!r}")
            branch_stack.append(curr_atom)
            i += 1
            continue

        if ch == ")":
            if not branch_stack:
                raise ValueError(f"Unbalanced ')' in SMILES: {s!r}")
            curr_atom = branch_stack.pop()
            pending_bond = None
            i += 1
            continue

        if ch == ".":
            curr_atom = -1
            pending_bond = None
            disconnected = True
            i += 1
            continue

        if ch in _BOND_CHARS:
            b_ord, b_ste = _BOND_CHARS[ch]
            pending_bond = (b_ord, b_ste)
            i += 1
            continue

        # Ring closure (%10..%99 or single digit 0..9)
        if ch == "%" or ch.isdigit():
            if curr_atom < 0:
                raise ValueError(f"Ring closure before any atom in SMILES: {s!r}")
            if ch == "%":
                if i + 2 >= n or not (s[i + 1].isdigit() and s[i + 2].isdigit()):
                    raise ValueError(f"Invalid two-digit ring closure at position {i} in SMILES: {s!r}")
                r_num = int(s[i + 1 : i + 3])
                i += 3
            else:
                r_num = int(ch)
                i += 1

            p_ord = pending_bond[0] if pending_bond is not None else None
            p_ste = pending_bond[1] if pending_bond is not None else BondStereo.NONE
            pending_bond = None

            if r_num not in ring_openings:
                ring_openings[r_num] = (curr_atom, p_ord, p_ste)
            else:
                open_atom, open_ord, open_ste = ring_openings.pop(r_num)
                if open_atom == curr_atom:
                    raise ValueError(f"Self-loop ring closure {r_num} on atom {curr_atom} in SMILES: {s!r}")
                resolved_ord = p_ord or open_ord
                if resolved_ord is None:
                    both_aro = bool(atom_specs[open_atom][5]) and bool(atom_specs[curr_atom][5])
                    resolved_ord = BondOrder.AROMATIC if both_aro else BondOrder.SINGLE
                resolved_ste = p_ste if p_ste != BondStereo.NONE else open_ste
                raw_edges.append([open_atom, curr_atom, resolved_ord, resolved_ste])
            continue

        # Bracketed atom [...]
        if ch == "[":
            close_idx = s.find("]", i + 1)
            if close_idx == -1:
                raise ValueError(f"Unclosed '[' at position {i} in SMILES: {s!r}")
            elem, iso, chi, h_cnt, chg, is_aro = _parse_bracket_atom(s[i + 1 : close_idx])
            new_idx = len(atom_specs)
            atom_specs.append([elem, iso, chi, h_cnt, chg, is_aro, True])
            i = close_idx + 1
        else:
            # Unbracketed organic-subset atom
            if i + 2 <= n and s[i : i + 2] in ("Cl", "Br"):
                sym = s[i : i + 2]
                elem = _ORGANIC_ALIPHATIC[sym]
                is_aro = False
                i += 2
            elif ch in _ORGANIC_ALIPHATIC:
                elem = _ORGANIC_ALIPHATIC[ch]
                is_aro = False
                i += 1
            elif ch in _ORGANIC_AROMATIC:
                elem = _ORGANIC_AROMATIC[ch]
                is_aro = True
                i += 1
            else:
                raise ValueError(f"Unexpected token {ch!r} at position {i} in SMILES: {s!r}")
            new_idx = len(atom_specs)
            atom_specs.append([elem, 0, AtomChirality.NONE, 0, 0, is_aro, False])

        if curr_atom >= 0 and not disconnected:
            if pending_bond is not None and pending_bond[0] is not None:
                b_ord = pending_bond[0]
                b_ste = pending_bond[1]
            else:
                both_aro = bool(atom_specs[curr_atom][5]) and bool(atom_specs[new_idx][5])
                b_ord = BondOrder.AROMATIC if both_aro else BondOrder.SINGLE
                b_ste = pending_bond[1] if pending_bond is not None else BondStereo.NONE
            raw_edges.append([curr_atom, new_idx, b_ord, b_ste])

        curr_atom = new_idx
        pending_bond = None
        disconnected = False

    if branch_stack:
        raise ValueError(f"Unclosed '(' in SMILES: {s!r}")
    if ring_openings:
        unclosed = sorted(ring_openings.keys())
        raise ValueError(f"Unclosed ring closures {unclosed} in SMILES: {s!r}")

    num_atoms = len(atom_specs)
    if num_atoms == 0:
        return (), (), b""

    # Compute bond valences, aromatic bond counts, and max bond order per atom
    bond_val = [0.0] * num_atoms
    aro_cnt = [0] * num_atoms
    max_ord = [0] * num_atoms
    double_cnt = [0] * num_atoms

    for u_r, v_r, b_ord_r, _ in raw_edges:
        u = int(u_r)
        v = int(v_r)
        bo = BondOrder(int(b_ord_r))
        if bo == BondOrder.AROMATIC:
            contrib = 1.5
            aro_cnt[u] += 1
            aro_cnt[v] += 1
        elif bo in (BondOrder.SINGLE, BondOrder.DOUBLE, BondOrder.TRIPLE):
            contrib = float(int(bo))
        else:
            contrib = 0.0
        bond_val[u] += contrib
        bond_val[v] += contrib
        bo_int = 2 if bo == BondOrder.AROMATIC else int(bo)
        if bo_int > max_ord[u]:
            max_ord[u] = bo_int
        if bo_int > max_ord[v]:
            max_ord[v] = bo_int
        if bo == BondOrder.DOUBLE:
            double_cnt[u] += 1
            double_cnt[v] += 1

    # Detect topological rings via DFS back-edges
    adj: list[list[tuple[int, int]]] = [[] for _ in range(num_atoms)]
    for k, (u_r, v_r, _, _) in enumerate(raw_edges):
        u, v = int(u_r), int(v_r)
        if u != v:
            adj[u].append((v, k))
            adj[v].append((u, k))

    depth: dict[int, int] = {}
    parent_node: dict[int, int] = {}
    parent_edge: dict[int, int] = {}
    ring_atoms: set[int] = set()
    ring_bonds: set[int] = set()

    for start in range(num_atoms):
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

    # Build final Atom tuple
    atoms_out: list[Atom] = []
    is_unsaturated = [False] * num_atoms
    for idx, spec in enumerate(atom_specs):
        elem = Element(int(spec[0]))
        iso = int(spec[1])
        chi = AtomChirality(int(spec[2]))
        h_cnt = int(spec[3])
        chg = int(spec[4])
        is_aro = bool(spec[5])
        is_bracket = bool(spec[6])

        if not is_bracket:
            h_cnt = _implicit_hydrogens(elem, is_aro, bond_val[idx], aro_cnt[idx])

        if max_ord[idx] == int(BondOrder.TRIPLE) or double_cnt[idx] >= 2:
            hyb = Hybridization.SP
            is_unsaturated[idx] = True
        elif is_aro or max_ord[idx] == int(BondOrder.DOUBLE):
            hyb = Hybridization.SP2
            is_unsaturated[idx] = True
        elif bond_val[idx] > 0 or h_cnt > 0:
            hyb = Hybridization.SP3
        else:
            hyb = Hybridization.UNSPECIFIED

        atoms_out.append(
            Atom(
                elem,
                charge=chg,
                hydrogens=min(15, h_cnt),
                chirality=chi,
                hybridization=hyb,
                is_aromatic=is_aro,
                in_ring=(idx in ring_atoms),
                isotope=iso,
            )
        )

    # Build final BondEdge tuple
    edges_out: list[BondEdge] = []
    for k, (u_r, v_r, b_ord_r, b_ste_r) in enumerate(raw_edges):
        u, v = int(u_r), int(v_r)
        bo = BondOrder(int(b_ord_r))
        bs = BondStereo(int(b_ste_r))
        conj = bo in (BondOrder.AROMATIC, BondOrder.DOUBLE, BondOrder.TRIPLE) or (
            is_unsaturated[u] and is_unsaturated[v]
        )
        bond = Bond(bo, stereo=bs, is_conjugated=conj, in_ring=(k in ring_bonds))
        edges_out.append(BondEdge(source=u, target=v, edge_type=bond))

    nodes_tuple = tuple(atoms_out)
    edges_tuple = tuple(edges_out)
    return _canonicalize_mol_graph(nodes_tuple, edges_tuple)


def _format_atom_token(atom: Atom, bond_valence: float, n_aro_bonds: int) -> str:
    r"""Format a single `Atom` as an unbracketed organic-subset token or a `[...]` bracketed token."""
    elem = atom.element
    sym = atom.symbol_str
    if atom.is_aromatic and sym.lower() in _ORGANIC_AROMATIC:
        sym_out = sym.lower()
    else:
        sym_out = sym

    can_unbracket = (
        (sym in _ORGANIC_ALIPHATIC or (atom.is_aromatic and sym.lower() in _ORGANIC_AROMATIC))
        and atom.isotope == 0
        and atom.charge == 0
        and atom.chirality == AtomChirality.NONE
        and atom.hydrogens == _implicit_hydrogens(elem, atom.is_aromatic, bond_valence, n_aro_bonds)
    )
    if can_unbracket:
        return sym_out

    parts: list[str] = ["["]
    if atom.isotope > 0:
        parts.append(str(atom.isotope))
    parts.append(sym.lower() if atom.is_aromatic else sym)
    if atom.chirality == AtomChirality.CCW:
        parts.append("@")
    elif atom.chirality == AtomChirality.CW:
        parts.append("@@")
    if atom.hydrogens == 1:
        parts.append("H")
    elif atom.hydrogens > 1:
        parts.append(f"H{atom.hydrogens}")
    if atom.charge > 0:
        parts.append("+" if atom.charge == 1 else f"+{atom.charge}")
    elif atom.charge < 0:
        parts.append("-" if atom.charge == -1 else f"{atom.charge}")
    parts.append("]")
    return "".join(parts)


def _format_bond_symbol(bond: Bond, u_aro: bool, v_aro: bool) -> str:
    r"""Format the bond symbol between two atoms given their aromaticity."""
    order = bond.order
    stereo = bond.stereo
    if stereo == BondStereo.UP:
        return "/"
    if stereo == BondStereo.DOWN:
        return "\\"
    if order == BondOrder.SINGLE:
        return "-" if (u_aro and v_aro) else ""
    if order == BondOrder.AROMATIC:
        return "" if (u_aro and v_aro) else ":"
    if order == BondOrder.DOUBLE:
        return "="
    if order == BondOrder.TRIPLE:
        return "#"
    if order == BondOrder.UNKNOWN:
        return "~"
    return ""


def _canonicalize_mol_graph(
    nodes: Sequence[Atom],
    edges: Sequence[BondEdge],
) -> tuple[tuple[Atom, ...], tuple[BondEdge, ...], bytes]:
    r"""Canonicalize `(nodes, edges)` into deterministic canonical DFS order and OpenSMILES byte-string."""
    n = len(nodes)
    if n == 0:
        return (), (), b""

    adj: list[list[tuple[int, Bond, int]]] = [[] for _ in range(n)]
    bond_val = [0.0] * n
    aro_cnt = [0] * n

    for k, e in enumerate(edges):
        u, v, b = e.source, e.target, e.edge_type
        if 0 <= u < n and 0 <= v < n and u != v:
            adj[u].append((v, b, k))
            adj[v].append((u, b, k))
            contrib = b.valence_contribution
            bond_val[u] += contrib
            bond_val[v] += contrib
            if b.is_aromatic:
                aro_cnt[u] += 1
                aro_cnt[v] += 1

    # Compute Morgan / Weisfeiler-Lehman canonical atom ranks (3 rounds of refinement)
    invariants: list[tuple[int, ...]] = [
        (
            int(a.element),
            a.isotope,
            a.charge,
            a.hydrogens,
            int(a.chirality),
            1 if a.is_aromatic else 0,
            1 if a.in_ring else 0,
            len(adj[idx]),
            int(round(bond_val[idx] * 2)),
        )
        for idx, a in enumerate(nodes)
    ]
    unique_inv = sorted(set(invariants))
    ranks = [unique_inv.index(inv) for inv in invariants]

    for _ in range(3):
        next_sigs: list[tuple[int, tuple[tuple[int, int], ...]]] = []
        for u in range(n):
            nbr_sig = tuple(sorted((int(b.order), ranks[v]) for v, b, _ in adj[u]))
            next_sigs.append((ranks[u], nbr_sig))
        unique_sigs = sorted(set(next_sigs))
        ranks = [unique_sigs.index(sig) for sig in next_sigs]

    atom_tokens = [_format_atom_token(nodes[u], bond_val[u], aro_cnt[u]) for u in range(n)]

    visited_atoms = [False] * n
    used_edges = [False] * len(edges)
    tree_children: dict[int, list[tuple[int, Bond]]] = {u: [] for u in range(n)}
    ring_closures_at: dict[int, list[tuple[int, str]]] = {u: [] for u in range(n)}
    dfs_order: dict[int, int] = {}
    comp_nodes: list[int] = []
    back_edges: list[tuple[int, int, Bond]] = []
    ring_open_for_close: dict[tuple[int, int], tuple[int, Bond]] = {}
    emitted_nodes: list[int] = []
    emitted_edges: list[tuple[int, int, Bond]] = []
    components: list[tuple[str, list[int], list[tuple[int, int, Bond]]]] = []

    def _build_dfs(u: int) -> None:
        visited_atoms[u] = True
        dfs_order[u] = len(comp_nodes)
        comp_nodes.append(u)
        nbrs = sorted(adj[u], key=lambda item: (ranks[item[0]], int(item[1].order), item[0]))
        for v, b, edge_k in nbrs:
            if used_edges[edge_k]:
                continue
            used_edges[edge_k] = True
            if not visited_atoms[v]:
                tree_children[u].append((v, b))
                _build_dfs(v)
            else:
                back_edges.append((u, v, b))

    def _emit_subtree(u: int) -> str:
        emitted_nodes.append(u)
        for r_idx, _ in ring_closures_at[u]:
            if (u, r_idx) in ring_open_for_close:
                open_u, b_ring = ring_open_for_close[(u, r_idx)]
                emitted_edges.append((open_u, u, b_ring))
        closures = "".join(tok for _, tok in ring_closures_at[u])
        base = atom_tokens[u] + closures
        children = tree_children[u]
        if not children:
            return base
        branch_parts: list[str] = []
        for v, b in children[:-1]:
            emitted_edges.append((u, v, b))
            b_sym = _format_bond_symbol(b, nodes[u].is_aromatic, nodes[v].is_aromatic)
            branch_parts.append(f"({b_sym}{_emit_subtree(v)})")
        last_v, last_b = children[-1]
        emitted_edges.append((u, last_v, last_b))
        last_sym = _format_bond_symbol(last_b, nodes[u].is_aromatic, nodes[last_v].is_aromatic)
        return base + "".join(branch_parts) + last_sym + _emit_subtree(last_v)

    while not all(visited_atoms):
        # Pick unvisited atom with smallest canonical rank as component root
        start = min((u for u in range(n) if not visited_atoms[u]), key=lambda u: (ranks[u], u))

        # Pass 1: Build canonical DFS spanning tree and identify ring-closure back-edges
        comp_nodes.clear()
        back_edges.clear()
        ring_open_for_close.clear()
        emitted_nodes.clear()
        emitted_edges.clear()
        _build_dfs(start)

        # Assign ring numbers 1, 2, ... in canonical order of their opening ancestor atom
        back_edges.sort(
            key=lambda item: (
                dfs_order[min(item[0], item[1], key=lambda x: dfs_order[x])],
                dfs_order[max(item[0], item[1], key=lambda x: dfs_order[x])],
            )
        )
        for r_idx, (u_back, v_back, b_ring) in enumerate(back_edges, start=1):
            r_str = str(r_idx) if r_idx < 10 else f"%{r_idx}"
            b_sym = _format_bond_symbol(b_ring, nodes[u_back].is_aromatic, nodes[v_back].is_aromatic)
            if dfs_order[v_back] < dfs_order[u_back]:
                open_u, close_u = v_back, u_back
            else:
                open_u, close_u = u_back, v_back
            ring_closures_at[open_u].append((r_idx, b_sym + r_str))
            ring_closures_at[close_u].append((r_idx, r_str))
            ring_open_for_close[(close_u, r_idx)] = (open_u, b_ring)

        comp_smi = _emit_subtree(start)
        components.append((comp_smi, list(emitted_nodes), list(emitted_edges)))

    components.sort(key=lambda item: item[0])
    canon_smi = ".".join(item[0] for item in components).encode("ascii")

    remap: dict[int, int] = {}
    canon_nodes_list: list[Atom] = []
    for _, c_nodes, _ in components:
        for old_u in c_nodes:
            remap[old_u] = len(canon_nodes_list)
            canon_nodes_list.append(nodes[old_u])

    canon_edges_list: list[BondEdge] = []
    for _, _, c_edges in components:
        for old_u, old_v, b in c_edges:
            canon_edges_list.append(BondEdge(source=remap[old_u], target=remap[old_v], edge_type=b))

    return tuple(canon_nodes_list), tuple(canon_edges_list), canon_smi


def format_smiles(nodes: Sequence[Atom], edges: Sequence[BondEdge]) -> bytes:
    r"""Serialize `(nodes, edges)` into a deterministic canonical OpenSMILES byte-string."""
    _, _, canon_smi = _canonicalize_mol_graph(nodes, edges)
    return canon_smi


# Streaming Reader & Writer --------------------------------------------------------------------------------------------
class SmilesReader(BaseReader[Mol, Mols]):
    r"""Streaming reader and Structure-of-Arrays batch loader for `.smi` / `.smiles` files.

    Supports plain and compressed (`.gz`, `.bz2`, `.xz`, `.zst`, `.zstd`) line-oriented SMILES
    streams where each non-empty, non-comment (`#`) line contains `SMILES` or `SMILES\tID`.
    """

    _EXTENSIONS: ClassVar[tuple[str, ...]] = ("smi", "smiles")
    __slots__ = ()

    def _iter_records(self) -> Iterator[Mol]:
        r"""Stream `Mol` records across chunk boundaries."""
        tail = b""
        while chunk := self._handle.read(self._chunk_size):
            buf = tail + chunk
            lines = buf.split(b"\n")
            tail = lines.pop()
            for raw_line in lines:
                line = raw_line.strip()
                if not line or line.startswith(b"#"):
                    continue
                yield Mol.from_smiles(line)
        if tail:
            line = tail.strip()
            if line and not line.startswith(b"#"):
                yield Mol.from_smiles(line)


class SmilesWriter(BaseWriter[Mol, Mols]):
    r"""Streaming writer and one-shot byte/file serializer for `.smi` / `.smiles` files."""

    _EXTENSIONS: ClassVar[tuple[str, ...]] = ("smi", "smiles")
    __slots__ = ()

    def _format_scalar(self, item: Mol) -> bytes:
        r"""Format a single `Mol` as its canonical SMILES line (`b"<smiles>\n"`)."""
        smi = item.canonical if item.canonical else item.to_smiles()
        return smi + b"\n"

    def _format_batch(self, batch: Mols) -> bytes:
        r"""Vectorized formatting of a `Mols` batch using stored `canonicals` when populated."""
        if len(batch) == 0:
            return b""
        lines: list[bytes] = []
        for i in range(len(batch)):
            c = bytes(batch.canonicals[i])
            lines.append((c if c else batch[i].to_smiles()) + b"\n")
        return b"".join(lines)
