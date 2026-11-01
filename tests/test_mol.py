"""Unit tests for `seroverse.core.mol` and `seroverse.core.io.smiles`."""

from pathlib import Path

import numpy as np
import pytest

from seroverse.core.io import SmilesReader, SmilesWriter
from seroverse.core.mol import (
    Atom,
    AtomChirality,
    Bond,
    BondOrder,
    BondStereo,
    Conformers,
    Element,
    Hybridization,
    Mol,
    Mols,
    Structure,
    Structures,
)


def test_periodic_table_elements() -> None:
    specs = Element.specs()
    assert len(specs) == 119
    assert Element.WILDCARD == 0
    assert Element.C == 6
    assert Element.OG == 118

    c_spec = Element.C.spec
    assert c_spec.name == b"C"
    assert c_spec.full_name == b"carbon"
    assert c_spec.atomic_number == 6
    assert c_spec.default_valence == 4
    assert pytest.approx(c_spec.mass, rel=1e-3) == 12.011

    assert Element.from_name("C") == Element.C
    assert Element.from_name("carbon") == Element.C
    assert Element.from_name(b"Fe") == Element.FE
    assert Element.from_name("iron") == Element.FE
    assert Element.from_name("*") == Element.WILDCARD

    fmat = specs.feature_matrix
    assert fmat.shape == (119, 6)
    assert fmat.dtype == np.float32
    assert not fmat.flags.writeable


def test_atom_and_bond_bit_packing() -> None:
    atom = Atom(
        Element.C,
        charge=-1,
        hydrogens=2,
        chirality=AtomChirality.CW,
        hybridization=Hybridization.SP3,
        is_aromatic=False,
        in_ring=True,
        isotope=13,
    )
    assert atom.element == Element.C
    assert atom.symbol == b"C"
    assert atom.charge == -1
    assert atom.hydrogens == 2
    assert atom.chirality == AtomChirality.CW
    assert atom.hybridization == Hybridization.SP3
    assert not atom.is_aromatic
    assert atom.in_ring
    assert atom.isotope == 13
    assert not atom.is_edge_node
    assert pytest.approx(atom.mass, rel=1e-3) == 13.0 + 2 * 1.008

    # Unpacking raw int preserves all bitfields
    unpacked = Atom(int(atom))
    assert unpacked == atom
    assert unpacked.charge == -1
    assert unpacked.isotope == 13

    bond = Bond(BondOrder.DOUBLE, stereo=BondStereo.UP, is_conjugated=True, in_ring=False)
    assert bond.order == BondOrder.DOUBLE
    assert bond.stereo == BondStereo.UP
    assert bond.is_conjugated
    assert not bond.in_ring
    assert bond.is_edge_node
    assert Bond(int(bond)) == bond


def test_smiles_parsing_and_canonicalization() -> None:
    # Ethanol: CCO vs OCC should canonicalize to the same SMILES and be isomorphic
    etoh1 = Mol.from_smiles("CCO")
    etoh2 = Mol.from_smiles("OCC")
    assert etoh1.canonical == etoh2.canonical
    assert etoh1.is_isomorphic(etoh2)
    assert etoh1.num_atoms == 3
    assert etoh1.num_bonds == 2
    assert etoh1.formula == "C2H6O"
    assert pytest.approx(etoh1.mass, rel=1e-3) == 46.069

    # Benzene: c1ccccc1
    bz = Mol.from_smiles("c1ccccc1")
    assert bz.num_atoms == 6
    assert bz.num_bonds == 6
    assert bz.formula == "C6H6"
    assert all(a.is_aromatic and a.in_ring and a.hydrogens == 1 for a in bz.nodes)
    assert all(e.edge_type.is_aromatic and e.edge_type.in_ring for e in bz.edges)

    # Bracketed atoms: [13C@H](F)(Cl)Br and [NH4+]
    chiral = Mol.from_smiles("[13C@H](F)(Cl)Br")
    c_atom = next(a for a in chiral.nodes if a.element == Element.C)
    assert c_atom.isotope == 13
    assert c_atom.chirality == AtomChirality.CCW
    assert c_atom.hydrogens == 1

    nh4 = Mol.from_smiles("[NH4+]")
    assert nh4.charge == 1
    assert nh4.formula == "H4N+"


def test_substructure_matching_and_wildcards() -> None:
    phenol = Mol.from_smiles("Oc1ccccc1")
    benzene = Mol.from_smiles("c1ccccc1")
    hydroxyl = Mol.from_smiles("CO")
    wildcard_aro = Mol.from_smiles("*c1ccccc1")
    pyridine = Mol.from_smiles("n1ccccc1")

    assert phenol.has_substruct_match(benzene)
    match = phenol.get_substruct_match(benzene)
    assert match is not None and len(match) == 6
    assert not phenol.has_substruct_match(hydroxyl)  # aromatic C vs aliphatic C
    assert phenol.has_substruct_match(wildcard_aro, allow_wildcards=True)
    assert not phenol.has_substruct_match(pyridine)

    batch = Mols.from_smiles(["CCO", "Oc1ccccc1", "c1ccccc1", "CC(=O)O"])
    hits = batch.has_substruct_match(benzene)
    assert hits.tolist() == [False, True, True, False]
    assert np.allclose(batch.masses, [m.mass for m in batch], rtol=1e-5)
    assert batch.charges.tolist() == [0, 0, 0, 0]
    assert batch[phenol.canonical] == phenol


def test_wwl_distance_zero_scipy() -> None:
    batch = Mols.from_smiles(["CCO", "CCCO", "c1ccccc1"])
    dmat = batch.wwl_distance(iterations=2)
    assert dmat.shape == (3, 3)
    assert np.allclose(np.diag(dmat), 0.0)
    assert np.allclose(dmat, dmat.T)
    # Ethanol (CCO) is chemically closer to propanol (CCCO) than to benzene (c1ccccc1)
    assert 0.0 < dmat[0, 1] < dmat[0, 2]
    assert pytest.approx(batch[0].wwl_distance(batch[1]), rel=1e-6) == dmat[0, 1]


def test_smiles_reader_writer_roundtrip(tmp_path: Path) -> None:
    smiles_in = ["CCO\tethanol", "c1ccccc1\tbenzene", "CC(=O)O\tacetic_acid"]
    raw_bytes = ("\n".join(smiles_in) + "\n").encode("ascii")

    mols = SmilesReader.from_bytes(raw_bytes).load()
    assert len(mols) == 3

    smi_path = tmp_path / "molecules.smi.gz"
    SmilesWriter.to_file(smi_path, mols)
    reloaded = SmilesReader.from_file(smi_path).load()
    assert reloaded == mols


def test_3d_conformer_and_structure() -> None:
    # Water-like O-H-H geometry in Angstroms
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
        dtype=np.float32,
    )
    struct = Structure.from_coords(["O", "H", "H"], coords, infer_bonds=True)
    assert struct.num_atoms == 3
    assert struct.num_bonds == 2  # Two O-H covalent bonds, H-H is > 2 * r_cov(H) + 0.45

    dmat = struct.distance_matrix()
    assert dmat.shape == (3, 3)
    assert pytest.approx(float(dmat[0, 1]), rel=1e-5) == 0.96

    contact = struct.contact_csr(cutoff=1.2)
    assert contact.shape == (3, 3)
    assert contact.nnz == 4  # 2 undirected O-H pairs = 4 directed entries

    # Rotate by 90 degrees around Z-axis and translate by (10, -5, 3)
    rot_coords = np.column_stack((-coords[:, 1] + 10.0, coords[:, 0] - 5.0, coords[:, 2] + 3.0))
    moved = Structure.from_coords(["O", "H", "H"], rot_coords, infer_bonds=True)
    assert pytest.approx(moved.rmsd_to(struct), abs=1e-5) == 0.0
    aligned = moved.align_to(struct)
    assert np.allclose(aligned.conformer.coords, struct.conformer.coords, atol=1e-4)

    # Batch Structures & Conformers
    structs = Structures.from_list([struct, moved])
    assert len(structs) == 2
    assert structs.centroids().shape == (2, 3)
    assert np.allclose(
        structs.radii_of_gyration(), [struct.radius_of_gyration(), moved.radius_of_gyration()], atol=1e-5
    )
    assert structs[0] == struct
    assert structs[1] == moved
    confs = Conformers.from_list([struct.conformer, moved.conformer])
    assert len(confs) == 2
