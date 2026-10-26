"""Unit tests for `seroverse.core.graph` (CSR, RelationalCSR, Graph, Graphs, Align, Arborescence, GNN, Layout)."""

from pathlib import Path

import numpy as np
import pytest

from seroverse.core.graph.align import (
    AlignmentConfig,
    DegreeProfileEmbedding,
    GraphAligner,
    IsomorphicGraphAligner,
    WeisfeilerLehmanEmbedding,
    _linear_sum_assignment_kernel,
)
from seroverse.core.graph.arborescence import Arborescence, Arborescences, ArborescenceSolver
from seroverse.core.graph.csr import CSR, RelationalCSR
from seroverse.core.graph.gnn import RelationalMPNN
from seroverse.core.graph.layout import LayoutEngine
from seroverse.core.graph.models import Graph, Graphs
from seroverse.core.seq.clustering import Clusterer
from seroverse.core.seq.seq import Sequences


def _make_sample_graphs() -> tuple[Graph, Graph, Graphs]:
    rng = np.random.default_rng(42)
    batch = Graphs.random(
        n_graphs=2,
        n_nodes=3,
        p=0.6,
        n_node_types=4,
        n_edge_types=2,
        directed=True,
        allow_self_loops=False,
        rng=rng,
    )
    return batch[0], batch[1], batch


def test_csr_core_operations() -> None:
    dense = np.array(
        [
            [0.0, 2.0, 0.0, 1.0],
            [2.0, 0.0, 3.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    csr = CSR.from_dense(dense)
    assert csr.shape == (4, 4)
    assert csr.nnz == 7
    np.testing.assert_allclose(csr.toarray(), dense)
    np.testing.assert_allclose(csr.T.toarray(), dense.T)

    # Row/col/total sums
    assert float(csr.sum()) == pytest.approx(float(dense.sum()))
    np.testing.assert_allclose(csr.sum(axis=1), dense.sum(axis=1))
    np.testing.assert_allclose(csr.sum(axis=0), dense.sum(axis=0))

    # SpMV & SpMM & CSR @ CSR
    vec = np.array([1.0, -1.0, 2.0, 0.5], dtype=np.float64)
    np.testing.assert_allclose(csr @ vec, dense @ vec)
    mat = np.arange(8, dtype=np.float64).reshape(4, 2)
    np.testing.assert_allclose(csr @ mat, dense @ mat)
    np.testing.assert_allclose((csr @ csr).toarray(), dense @ dense)
    np.testing.assert_allclose((csr + csr).toarray(), dense + dense)

    # Submatrix slicing
    sub = csr[[0, 2, 3], [0, 2, 3]]
    np.testing.assert_allclose(sub.toarray(), dense[np.ix_([0, 2, 3], [0, 2, 3])])

    # Laplacian & connected components
    lap = csr.laplacian(normed=True)
    assert lap.shape == (4, 4)
    n_comp, labels = csr.connected_components()
    assert n_comp == 1
    assert len(labels) == 4

    # with_diag & without_zeros
    csr_no_diag = csr.with_diag(0).without_zeros()
    assert csr_no_diag.nnz == 6
    np.testing.assert_allclose(np.diag(csr_no_diag.toarray()), 0.0)


def test_relational_csr_hierarchy_and_builders() -> None:
    rel_csr = RelationalCSR.from_multihop(
        node_offsets=[0, 3, 5],
        edge_sources=[0, 1, 3],
        edge_targets=[1, 2, 4],
        edge_orientations=[0, 1, 0],
        include_2hop=True,
        add_virtual_pool=True,
    )
    assert isinstance(rel_csr, CSR)
    assert rel_csr.n_nodes == 7
    assert rel_csr.n_relations == 8
    assert len(rel_csr.virtual_nodes) == 2
    assert len(rel_csr.src_indices) == rel_csr.nnz
    assert len(rel_csr.edge_weights) == rel_csr.nnz

    g0, _, batch = _make_sample_graphs()
    g_rcsr = g0.as_relational_csr()
    b_rcsr = batch.as_relational_csr()
    assert isinstance(g_rcsr, RelationalCSR)
    assert isinstance(b_rcsr, RelationalCSR)
    assert g_rcsr.n_nodes == 4  # 3 real + 1 virtual pool
    assert b_rcsr.n_nodes == 8  # 6 real + 2 virtual pool


def test_graph_models_and_spectral_distance() -> None:
    g0, g1, batch = _make_sample_graphs()
    adj0, labels0 = g0.as_graph(as_bipartite=False)
    assert isinstance(adj0, CSR)
    assert len(labels0) == 3

    pan_adj, pan_nodes, pan_counts = batch.as_pan_graph()
    assert isinstance(pan_adj, CSR)
    assert len(pan_nodes) == len(pan_counts)
    assert int(pan_counts.sum()) == batch.total_nodes

    block_adj = batch.get_block_diagonal_graph()
    assert isinstance(block_adj, CSR)
    assert block_adj.shape == (batch.total_nodes, batch.total_nodes)

    d_self = g0.spectral_distance(g0)
    d_diff = g0.spectral_distance(g1)
    assert d_self == pytest.approx(0.0, abs=1e-12)
    assert d_diff >= 0.0

    dist_mat = batch.spectral_distance_matrix()
    assert dist_mat.shape == (2, 2)
    np.testing.assert_allclose(np.diag(dist_mat), 0.0, atol=1e-12)
    assert dist_mat[0, 1] == pytest.approx(dist_mat[1, 0])

    # Erdős–Rényi random batch generation (variable node counts, undirected, slicing & round-trip)
    er_batch = Graphs.random(
        n_graphs=6,
        n_nodes=(4, 8),
        p=0.35,
        n_node_types=5,
        n_edge_types=3,
        directed=False,
        allow_self_loops=False,
        rng=np.random.default_rng(123),
    )
    assert len(er_batch) == 6
    assert np.all((er_batch.node_lengths >= 4) & (er_batch.node_lengths <= 8))
    assert np.all(er_batch.edge_sources != er_batch.edge_targets)
    rebuilt = Graphs.from_list(list(er_batch))
    np.testing.assert_array_equal(rebuilt.node_types, er_batch.node_types)
    np.testing.assert_array_equal(rebuilt.edge_sources, er_batch.edge_sources)
    np.testing.assert_array_equal(rebuilt.edge_targets, er_batch.edge_targets)


def test_linear_sum_assignment_and_graph_aligner() -> None:
    # Rectangular nr < nc
    cost_wide = np.array([[4.0, 1.0, 3.0], [6.0, 0.0, 5.0]], dtype=np.float64)
    r_w, c_w = _linear_sum_assignment_kernel(cost_wide)
    np.testing.assert_array_equal(r_w, [0, 1])
    np.testing.assert_array_equal(c_w, [2, 1])

    # Rectangular nr > nc
    r_t, c_t = _linear_sum_assignment_kernel(cost_wide.T)
    np.testing.assert_array_equal(r_t, [1, 2])
    np.testing.assert_array_equal(c_t, [1, 0])

    g0, g1, _ = _make_sample_graphs()
    aligner = GraphAligner(embedding_strategy=WeisfeilerLehmanEmbedding(iterations=2), config=AlignmentConfig.default())
    res = aligner.align(g0, g1)
    assert len(res.matches) + len(res.unmatched_a) == len(g0.nodes) + len(g0.edges)

    deg_aligner = GraphAligner(embedding_strategy=DegreeProfileEmbedding(), config=AlignmentConfig.rigid())
    res_deg = deg_aligner.align(g0, g0)
    assert len(res_deg.unmatched_a) == 0

    iso_aligner = IsomorphicGraphAligner()
    res_iso = iso_aligner.align(g0, g0)
    assert res_iso.total_cost == pytest.approx(0.0, abs=1e-9)


def test_arborescence_scalar_container_and_solver() -> None:
    weights_2d = np.array(
        [
            [0.0, 5.0, 1.0, 0.5],
            [0.0, 0.0, 4.0, 2.0],
            [0.0, 1.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    arb = Arborescence.maximum(weights_2d, root=0)
    assert arb.score == pytest.approx(15.0)
    np.testing.assert_array_equal(arb.parents, [-1, 0, 1, 2])
    assert arb.edges.shape == (3, 3)

    costs_2d = np.where(weights_2d > 0, 10.0 - weights_2d, np.inf)
    min_arb = Arborescence.minimum(costs_2d, root=0)
    assert min_arb.score == pytest.approx(15.0)

    # 3D with slot exclusivity and k-best container
    weights_3d = np.zeros((3, 3, 2), dtype=np.float64) - 100.0
    weights_3d[0, 1, 0] = 10.0
    weights_3d[0, 2, 0] = 9.0  # Competes for slot 0 on parent 0
    weights_3d[0, 2, 1] = 7.0  # Uses slot 1 on parent 0
    weights_3d[1, 2, 0] = 8.0

    solver = ArborescenceSolver(mode="max", root=0, k=3, slot_exclusivity=True)
    arbs = solver.solve_k_best(weights_3d)
    assert isinstance(arbs, Arborescences)
    assert len(arbs) >= 2
    assert arbs[0].score == pytest.approx(18.0)  # 0->1 (10) + 1->2 (8)
    assert arbs[1].score == pytest.approx(17.0)  # 0->1 (10, slot 0) + 0->2 (7, slot 1)

    # Ragged slicing & concatenation on Arborescences
    sliced = arbs[:2]
    assert isinstance(sliced, Arborescences)
    assert len(sliced) == 2
    masked = arbs[np.array([True, False, True][: len(arbs)])]
    assert len(masked) >= 1
    combined = Arborescences.concat([sliced, masked])
    assert len(combined) == len(sliced) + len(masked)


def test_relational_mpnn_forward_backward_and_io(tmp_path: Path) -> None:
    _, _, batch = _make_sample_graphs()
    csr = batch.as_relational_csr()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 8)).astype(np.float32)
    node_targets = np.array([0, 1, 2, 0, 1, 3], dtype=np.int32)
    edge_indices = np.array([0, 1, 2], dtype=np.int32)
    parent_targets = np.array([-1, 0, 1], dtype=np.int32)
    slot_targets = np.array([-1, 0, 1], dtype=np.int32)
    clamped = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)

    model = RelationalMPNN(
        in_dim=8,
        hidden_dim=16,
        num_node_classes=4,
        num_edge_channels=3,
        num_relations=8,
        num_layers=2,
        bilinear_rank=8,
        learning_rate=5e-3,
        random_state=42,
    )
    loss0 = model.fit_step(
        x,
        csr,
        node_targets=node_targets,
        edge_node_indices=edge_indices,
        edge_parent_targets=parent_targets,
        edge_slot_targets=slot_targets,
        clamped_mask=clamped,
    )
    loss1 = model.fit_step(
        x,
        batch,
        node_targets=node_targets,
        edge_node_indices=edge_indices,
        edge_parent_targets=parent_targets,
        edge_slot_targets=slot_targets,
        clamped_mask=clamped,
    )
    assert np.isfinite(loss0) and np.isfinite(loss1)
    assert loss1 < loss0

    proba = model.predict_node_proba(x, batch)
    assert proba.shape == (8, 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    edge_scores = model.score_edges(model.encode(x, csr), edge_indices)
    assert edge_scores.shape == (3, 3, 3)

    save_path = model.save(tmp_path / "mpnn_test.npz")
    loaded = RelationalMPNN.load(save_path)
    np.testing.assert_allclose(loaded.predict_node_logits(x, csr), model.predict_node_logits(x, csr), atol=1e-6)


def test_layout_engine_and_clustering_csr() -> None:
    g0, _, batch = _make_sample_graphs()
    for init_mode in ("random", "circular", "spectral"):
        engine = LayoutEngine(dim=2, iterations=15, init=init_mode, seed=7)
        coords = engine(g0)
        assert coords.shape == (3, 2)
        assert np.all(np.isfinite(coords))

    batch_coords = LayoutEngine(dim=3, iterations=10, seed=7)(batch)
    _, pan_nodes, _ = batch.as_pan_graph()
    assert batch_coords.shape == (len(pan_nodes), 3)

    # Sequence clustering modes using CSR & connected_components_from_edges
    seqs = Sequences.from_bytes(
        [
            b"MKKVLLVTGAGGSIGSELCRQILK",
            b"MKKVLLVTGAGGSIGSELCRQILK",
            b"MKKVLLVTGAGGSIGSELCRQILR",
            b"MNTQIIIFGAAAGVGKTTLLRRLA",
            b"MNTQIIIFGAAAGVGKTTLLRRLA",
        ],
    )
    for mode in ("cdhit", "connected_components", "upgma", "mcl", "greedy_cover"):
        clusters = Clusterer(min_identity=80.0, mode=mode).cluster(seqs)
        csr_mem = clusters.as_csr(n_seqs=len(seqs))
        assert isinstance(csr_mem, CSR)
        assert csr_mem.shape == (len(clusters), len(seqs))

    n_c, comp_l = CSR.connected_components_from_edges(
        4, np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)
    )
    assert n_c == 2
    assert comp_l[0] == comp_l[1] and comp_l[2] == comp_l[3] and comp_l[0] != comp_l[2]
