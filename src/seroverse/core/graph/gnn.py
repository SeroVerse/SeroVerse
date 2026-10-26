r"""Numba-accelerated Relational Message-Passing Neural Network ([`RelationalMPNN`][seroverse.core.graph.gnn.RelationalMPNN]).

Provides compiled `@njit(parallel=True, fastmath=True)` message-passing forward/backward kernels,
factorized bilinear directed edge/slot scorers, and the [`RelationalMPNN`][seroverse.core.graph.gnn.RelationalMPNN]
model operating over [`RelationalCSR`][seroverse.core.graph.csr.RelationalCSR] graphs.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
from numba import njit, prange

from seroverse.core.graph.csr import RelationalCSR
from seroverse.core.graph.models import Graph, Graphs


# ----------------------------------------------------------------------------------------------------------------------
# Private Compiled Numba Kernels: R-MPNN Forward, Backward, Bilinear Edge Scorer, AdamW
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _mpnn_layer_forward(
    h_in: np.ndarray,  # (N, D)
    indptr: np.ndarray,  # (N + 1,)
    src_indices: np.ndarray,  # (E,)
    rel_types: np.ndarray,  # (E,)
    edge_weights: np.ndarray,  # (E,)
    w_rel: np.ndarray,  # (R, D, D)
    b_rel: np.ndarray,  # (D,)
    clamped_mask: np.ndarray,  # (N,) uint8
    clamped_anchor: np.ndarray,  # (N, D)
    norm_eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Parallel relational message-passing forward kernel with per-relation aggregation, ReLU, and RMSNorm."""
    n_nodes, dim = h_in.shape
    n_rel = w_rel.shape[0]
    h_out = np.empty((n_nodes, dim), dtype=np.float32)
    pre_norm = np.empty((n_nodes, dim), dtype=np.float32)
    rms_inv = np.empty(n_nodes, dtype=np.float32)
    eps = np.float32(norm_eps)
    inv_d = np.float32(1.0 / max(dim, 1))

    for v in prange(n_nodes):
        msg = np.empty(dim, dtype=np.float32)
        for d in range(dim):
            msg[d] = b_rel[d]

        start = int(indptr[v])
        end = int(indptr[v + 1])
        if end > start:
            agg_rel = np.zeros((n_rel, dim), dtype=np.float32)
            active_rel = np.zeros(n_rel, dtype=np.uint8)
            for k in range(start, end):
                u = int(src_indices[k])
                r = int(rel_types[k])
                w_e = edge_weights[k]
                active_rel[r] = 1
                for d_in in range(dim):
                    agg_rel[r, d_in] += w_e * h_in[u, d_in]

            for r in range(n_rel):
                if active_rel[r] != 0:
                    for d_in in range(dim):
                        hu_val = agg_rel[r, d_in]
                        for d_out in range(dim):
                            msg[d_out] += hu_val * w_rel[r, d_in, d_out]

        # ReLU activation on aggregated relational message + residual connection
        sq_sum = np.float32(0.0)
        is_clamped = clamped_mask[v] != 0
        for d in range(dim):
            act = msg[d] if msg[d] > np.float32(0.0) else np.float32(0.0)
            res = h_in[v, d] + act
            if is_clamped:
                res = np.float32(0.5) * res + np.float32(0.5) * clamped_anchor[v, d]
            pre_norm[v, d] = res
            sq_sum += res * res

        inv_rms = np.float32(1.0) / np.sqrt(sq_sum * inv_d + eps)
        rms_inv[v] = inv_rms
        for d in range(dim):
            h_out[v, d] = pre_norm[v, d] * inv_rms

    return h_out, pre_norm, rms_inv


@njit(cache=True, nogil=True, fastmath=True)
def _mpnn_layer_backward(
    d_h_out: np.ndarray,  # (N, D)
    h_in: np.ndarray,  # (N, D)
    pre_norm: np.ndarray,  # (N, D)
    rms_inv: np.ndarray,  # (N,)
    indptr: np.ndarray,  # (N + 1,)
    src_indices: np.ndarray,  # (E,)
    rel_types: np.ndarray,  # (E,)
    edge_weights: np.ndarray,  # (E,)
    w_rel: np.ndarray,  # (R, D, D)
    clamped_mask: np.ndarray,  # (N,) uint8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Backward pass through RMSNorm, clamped residual anchor, ReLU, and per-relation CSR aggregation."""
    n_nodes, dim = h_in.shape
    n_rel = w_rel.shape[0]
    d_h_in = np.zeros((n_nodes, dim), dtype=np.float32)
    d_w_rel = np.zeros((n_rel, dim, dim), dtype=np.float32)
    d_b_rel = np.zeros(dim, dtype=np.float32)
    inv_d = np.float32(1.0 / max(dim, 1))

    agg_rel = np.empty((n_rel, dim), dtype=np.float32)
    grad_rel = np.empty((n_rel, dim), dtype=np.float32)
    active_rel = np.zeros(n_rel, dtype=np.uint8)

    for v in range(n_nodes):
        inv_rms = rms_inv[v]
        inv_rms3 = inv_rms * inv_rms * inv_rms * inv_d

        dot_dy_x = np.float32(0.0)
        for d in range(dim):
            dot_dy_x += d_h_out[v, d] * pre_norm[v, d]

        d_res = np.empty(dim, dtype=np.float32)
        is_clamped = clamped_mask[v] != 0
        scale_c = np.float32(0.5) if is_clamped else np.float32(1.0)
        for d in range(dim):
            dr = (d_h_out[v, d] * inv_rms - pre_norm[v, d] * inv_rms3 * dot_dy_x) * scale_c
            d_res[d] = dr
            d_h_in[v, d] += dr

        d_msg = np.empty(dim, dtype=np.float32)
        any_active = False
        for d in range(dim):
            raw_res = (np.float32(2.0) * pre_norm[v, d]) if is_clamped else pre_norm[v, d]
            act = raw_res - h_in[v, d]
            if act > np.float32(1e-7):
                d_msg[d] = d_res[d]
                d_b_rel[d] += d_res[d]
                any_active = True
            else:
                d_msg[d] = np.float32(0.0)

        if not any_active:
            continue

        start = int(indptr[v])
        end = int(indptr[v + 1])
        if end <= start:
            continue

        for r in range(n_rel):
            active_rel[r] = 0

        for k in range(start, end):
            u = int(src_indices[k])
            r = int(rel_types[k])
            w_e = edge_weights[k]
            if active_rel[r] == 0:
                active_rel[r] = 1
                for d_in in range(dim):
                    agg_rel[r, d_in] = np.float32(0.0)
                    g_val = np.float32(0.0)
                    for d_out in range(dim):
                        g_val += w_rel[r, d_in, d_out] * d_msg[d_out]
                    grad_rel[r, d_in] = g_val

            for d_in in range(dim):
                agg_rel[r, d_in] += w_e * h_in[u, d_in]
                d_h_in[u, d_in] += w_e * grad_rel[r, d_in]

        for r in range(n_rel):
            if active_rel[r] != 0:
                for d_in in range(dim):
                    ar = agg_rel[r, d_in]
                    for d_out in range(dim):
                        dm = d_msg[d_out]
                        if dm != np.float32(0.0):
                            d_w_rel[r, d_in, d_out] += ar * dm

    return d_h_in, d_w_rel, d_b_rel


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _bilinear_edge_scores_forward(
    u_proj: np.ndarray,  # (M, K)
    v_proj: np.ndarray,  # (M, K)
    w_chan: np.ndarray,  # (C, K)
    b_chan: np.ndarray,  # (C,)
) -> np.ndarray:
    r"""Compute factorized bilinear multi-channel edge logits `S[u, v, c] = sum_k u_proj[u, k] v_proj[v, k] W_chan[c, k] / sqrt(K) + b_c`."""
    m_nodes = u_proj.shape[0]
    n_chan, rank_k = w_chan.shape
    scores = np.empty((m_nodes, m_nodes, n_chan), dtype=np.float32)
    inv_sqrt_k = np.float32(1.0 / np.sqrt(float(max(rank_k, 1))))

    for u in prange(m_nodes):
        for v in range(m_nodes):
            if u == v:
                for c in range(n_chan):
                    scores[u, v, c] = np.float32(-1e9)
                continue
            for c in range(n_chan):
                s = b_chan[c]
                for k in range(rank_k):
                    s += u_proj[u, k] * v_proj[v, k] * w_chan[c, k] * inv_sqrt_k
                scores[u, v, c] = s

    return scores


@njit(cache=True, nogil=True, fastmath=True)
def _bilinear_edge_scores_backward(
    d_scores: np.ndarray,  # (M, M, C)
    u_proj: np.ndarray,  # (M, K)
    v_proj: np.ndarray,  # (M, K)
    w_chan: np.ndarray,  # (C, K)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Backward pass for factorized bilinear multi-channel tensor contraction."""
    m_nodes = u_proj.shape[0]
    n_chan, rank_k = w_chan.shape
    inv_sqrt_k = np.float32(1.0 / np.sqrt(float(max(rank_k, 1))))

    d_u_proj = np.zeros((m_nodes, rank_k), dtype=np.float32)
    d_v_proj = np.zeros((m_nodes, rank_k), dtype=np.float32)
    d_w_chan = np.zeros((n_chan, rank_k), dtype=np.float32)
    d_b_chan = np.zeros(n_chan, dtype=np.float32)

    for u in range(m_nodes):
        for v in range(m_nodes):
            if u == v:
                continue
            for c in range(n_chan):
                ds = d_scores[u, v, c]
                if ds == np.float32(0.0):
                    continue
                d_b_chan[c] += ds
                ds_scaled = ds * inv_sqrt_k
                for k in range(rank_k):
                    uv_k = u_proj[u, k] * v_proj[v, k]
                    d_w_chan[c, k] += ds_scaled * uv_k
                    wc = w_chan[c, k] * ds_scaled
                    d_u_proj[u, k] += wc * v_proj[v, k]
                    d_v_proj[v, k] += wc * u_proj[u, k]

    return d_u_proj, d_v_proj, d_w_chan, d_b_chan


@njit(cache=True, nogil=True, fastmath=True)
def _adamw_step(
    param: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    step_size: float,
    wd_factor: float,
    apply_wd: bool,
) -> None:
    r"""In-place 1D/raveled AdamW update kernel for arbitrary-rank contiguous parameter tensors."""
    beta1 = np.float32(0.9)
    beta2 = np.float32(0.999)
    eps = np.float32(1e-8)
    ss = np.float32(step_size)
    wf = np.float32(wd_factor)
    p_flat = param.ravel()
    g_flat = grad.ravel()
    m_flat = m.ravel()
    v_flat = v.ravel()
    for i in range(len(p_flat)):
        g = g_flat[i]
        m_flat[i] = beta1 * m_flat[i] + (np.float32(1.0) - beta1) * g
        v_flat[i] = beta2 * v_flat[i] + (np.float32(1.0) - beta2) * g * g
        if apply_wd:
            p_flat[i] = p_flat[i] * wf - ss * m_flat[i] / (np.sqrt(v_flat[i]) + eps)
        else:
            p_flat[i] -= ss * m_flat[i] / (np.sqrt(v_flat[i]) + eps)


# ----------------------------------------------------------------------------------------------------------------------
# High-Level RelationalMPNN Model
# ----------------------------------------------------------------------------------------------------------------------
class RelationalMPNN:
    r"""Numba-powered Relational Message-Passing Neural Network (R-MPNN) for node & directed tree inference.

    Architecture:
        1. Input projection: $h^{(0)} = \text{RMSNorm}(X W_{\text{in}} + b_{\text{in}})$.
        2. $L$ Relational Message-Passing layers with residual connections, ReLU, RMSNorm, and
           optional clamped-node anchor injection (`clamped_mask`) so pre-existing annotations
           anchor contextual propagation without drifting.
        3. Node readout head: $Z_{\text{node}} = h^{(L)} W_{\text{node}} + b_{\text{node}}$ (`(N, num_node_classes)`).
        4. Factorized bilinear multi-slot directed edge readout head:
           $S[u, v, c] = \frac{1}{\sqrt{K}} \sum_{k=1}^K (h_u^{(L)} W_{\text{src}})_k (h_v^{(L)} W_{\text{dst}})_k W_{\text{chan}}[c, k] + b_{\text{chan}}[c]$
           producing `(M, M, num_edge_channels)` logits suitable for decoding via
           [`ArborescenceSolver`][seroverse.core.graph.arborescence.ArborescenceSolver] or
           [`Arborescence.maximum`][seroverse.core.graph.arborescence.Arborescence.maximum].
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_node_classes: int = 32,
        num_edge_channels: int = 8,
        num_relations: int = 8,
        num_layers: int = 3,
        bilinear_rank: int = 32,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        random_state: int | None = 42,
    ) -> None:
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_node_classes = int(num_node_classes)
        self.num_edge_channels = int(num_edge_channels)
        self.num_relations = int(num_relations)
        self.num_layers = int(num_layers)
        self.bilinear_rank = int(bilinear_rank)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_state = random_state

        self._step = 0
        self.loss_history_: list[float] = []
        self._init_weights()

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.random_state)
        d_in = self.in_dim
        d_hid = self.hidden_dim
        n_rel = self.num_relations
        n_lay = self.num_layers
        n_cls = self.num_node_classes
        n_chan = self.num_edge_channels
        rank_k = self.bilinear_rank

        self.w_in_ = rng.normal(0.0, np.sqrt(2.0 / max(d_in, 1)), size=(d_in, d_hid)).astype(np.float32)
        self.b_in_ = np.zeros(d_hid, dtype=np.float32)

        self.w_rel_ = rng.normal(
            0.0,
            np.sqrt(1.0 / max(d_hid * n_rel, 1)),
            size=(n_lay, n_rel, d_hid, d_hid),
        ).astype(np.float32)
        self.b_rel_ = np.zeros((n_lay, d_hid), dtype=np.float32)

        self.w_node_ = rng.normal(0.0, np.sqrt(1.0 / max(d_hid, 1)), size=(d_hid, n_cls)).astype(np.float32)
        self.b_node_ = np.zeros(n_cls, dtype=np.float32)

        self.w_edge_src_ = rng.normal(0.0, np.sqrt(1.0 / max(d_hid, 1)), size=(d_hid, rank_k)).astype(np.float32)
        self.w_edge_dst_ = rng.normal(0.0, np.sqrt(1.0 / max(d_hid, 1)), size=(d_hid, rank_k)).astype(np.float32)
        self.w_edge_chan_ = rng.normal(0.0, np.sqrt(1.0 / max(rank_k, 1)), size=(n_chan, rank_k)).astype(np.float32)
        self.b_edge_chan_ = np.zeros(n_chan, dtype=np.float32)

        self._m_w_in = np.zeros_like(self.w_in_)
        self._v_w_in = np.zeros_like(self.w_in_)
        self._m_b_in = np.zeros_like(self.b_in_)
        self._v_b_in = np.zeros_like(self.b_in_)

        self._m_w_rel = np.zeros_like(self.w_rel_)
        self._v_w_rel = np.zeros_like(self.w_rel_)
        self._m_b_rel = np.zeros_like(self.b_rel_)
        self._v_b_rel = np.zeros_like(self.b_rel_)

        self._m_w_node = np.zeros_like(self.w_node_)
        self._v_w_node = np.zeros_like(self.w_node_)
        self._m_b_node = np.zeros_like(self.b_node_)
        self._v_b_node = np.zeros_like(self.b_node_)

        self._m_w_esrc = np.zeros_like(self.w_edge_src_)
        self._v_w_esrc = np.zeros_like(self.w_edge_src_)
        self._m_w_edst = np.zeros_like(self.w_edge_dst_)
        self._v_w_edst = np.zeros_like(self.w_edge_dst_)
        self._m_w_echan = np.zeros_like(self.w_edge_chan_)
        self._v_w_echan = np.zeros_like(self.w_edge_chan_)
        self._m_b_echan = np.zeros_like(self.b_edge_chan_)
        self._v_b_echan = np.zeros_like(self.b_edge_chan_)

    @staticmethod
    def _coerce_csr(csr: RelationalCSR | Graph | Graphs[Any, Any]) -> RelationalCSR:
        """Coerce a `GraphModel` or `GraphBatch` into a `RelationalCSR` if needed."""
        if isinstance(csr, RelationalCSR):
            return csr
        if isinstance(csr, (Graph, Graphs)):
            return csr.as_relational_csr()
        raise TypeError(f"Expected RelationalCSR, GraphModel, or GraphBatch; got {type(csr).__name__}.")

    def _prepare_features(
        self,
        x: np.ndarray,
        csr: RelationalCSR,
        clamped_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad virtual pool rows if `x` only has real vertices, and normalize `clamped_mask`."""
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.in_dim:
            raise ValueError(f"Expected 2D input features of width {self.in_dim}; got {x_arr.shape}.")
        n_total = csr.n_nodes
        if x_arr.shape[0] < n_total:
            pad = np.zeros((n_total - x_arr.shape[0], self.in_dim), dtype=np.float32)
            x_full = np.ascontiguousarray(np.vstack([x_arr, pad]))
        elif x_arr.shape[0] == n_total:
            x_full = np.ascontiguousarray(x_arr)
        else:
            raise ValueError(f"x has {x_arr.shape[0]} rows, exceeding csr.n_nodes ({n_total}).")

        if clamped_mask is None:
            c_mask = np.zeros(n_total, dtype=np.uint8)
        else:
            cm = np.asarray(clamped_mask, dtype=np.uint8)
            if len(cm) < n_total:
                c_mask = np.zeros(n_total, dtype=np.uint8)
                c_mask[: len(cm)] = cm
            else:
                c_mask = np.ascontiguousarray(cm[:n_total])
        return x_full, c_mask

    def encode(
        self,
        x: np.ndarray,
        csr: RelationalCSR | Graph | Graphs[Any, Any],
        *,
        clamped_mask: np.ndarray | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        r"""Run the $L$-layer R-MPNN encoder and return final node embeddings `(N, hidden_dim)`."""
        rel_csr = self._coerce_csr(csr)
        x_full, c_mask = self._prepare_features(x, rel_csr, clamped_mask)
        h0 = np.ascontiguousarray(x_full @ self.w_in_ + self.b_in_[None, :], dtype=np.float32)
        h = h0
        for lay in range(self.num_layers):
            h, _, _ = _mpnn_layer_forward(
                h,
                rel_csr.indptr,
                rel_csr.src_indices,
                rel_csr.rel_types,
                rel_csr.edge_weights,
                self.w_rel_[lay],
                self.b_rel_[lay],
                c_mask,
                h0,
            )
        return h

    def predict_node_logits(
        self,
        x: np.ndarray,
        csr: RelationalCSR | Graph | Graphs[Any, Any],
        *,
        clamped_mask: np.ndarray | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        r"""Compute pre-softmax node class logits `(N, num_node_classes)`."""
        h = self.encode(x, csr, clamped_mask=clamped_mask)
        return np.ascontiguousarray(h @ self.w_node_ + self.b_node_[None, :], dtype=np.float32)

    def predict_node_proba(
        self,
        x: np.ndarray,
        csr: RelationalCSR | Graph | Graphs[Any, Any],
        *,
        clamped_mask: np.ndarray | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        r"""Compute softmax node class probabilities `(N, num_node_classes)`."""
        logits = self.predict_node_logits(x, csr, clamped_mask=clamped_mask)
        max_l = np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        return (exp_l / np.maximum(np.sum(exp_l, axis=1, keepdims=True), np.float32(1e-12))).astype(np.float32)

    def score_edges(
        self,
        h_nodes: np.ndarray,
        node_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float32]]:
        r"""Score all directed candidate edges `(u -> v, channel)` among `h_nodes` (or `h_nodes[node_indices]`).

        Returns:
            np.ndarray: 3D `float32` tensor of shape `(M, M, num_edge_channels)`.
        """
        h_sub = (
            np.ascontiguousarray(h_nodes[np.asarray(node_indices, dtype=np.int32)], dtype=np.float32)
            if node_indices is not None
            else np.ascontiguousarray(h_nodes, dtype=np.float32)
        )
        u_proj = np.ascontiguousarray(h_sub @ self.w_edge_src_, dtype=np.float32)
        v_proj = np.ascontiguousarray(h_sub @ self.w_edge_dst_, dtype=np.float32)
        return _bilinear_edge_scores_forward(u_proj, v_proj, self.w_edge_chan_, self.b_edge_chan_)

    def fit_step(
        self,
        x: np.ndarray,
        csr: RelationalCSR | Graph | Graphs[Any, Any],
        *,
        node_targets: np.ndarray | None = None,
        node_weights: np.ndarray | None = None,
        edge_node_indices: np.ndarray | None = None,
        edge_parent_targets: np.ndarray | None = None,
        edge_slot_targets: np.ndarray | None = None,
        clamped_mask: np.ndarray | None = None,
        lr_scale: float = 1.0,
    ) -> float:
        r"""Execute one complete forward + backward + AdamW optimization step on a graph or batch.

        Args:
            x: Input node feature matrix `(N_real, in_dim)` or `(N_total, in_dim)`.
            csr: [`RelationalCSR`][seroverse.core.graph.csr.RelationalCSR], [`GraphModel`][seroverse.core.graph.models.GraphModel],
                or [`GraphBatch`][seroverse.core.graph.models.GraphBatch] connectivity structure.
            node_targets: Optional 1D `int32` array of ground-truth node class indices (`-1` = ignore).
            node_weights: Optional 1D `float32` array of per-node loss weights.
            edge_node_indices: Optional 1D `int32` array of active tree vertices `[v_0, ..., v_{M-1}]`.
            edge_parent_targets: Optional 1D `int32` array of length `M` giving target parent index in `[0, M-1]`
                for each active tree vertex (`-1` for root or unsupervised vertices).
            edge_slot_targets: Optional 1D `int32` array of length `M` giving target slot/channel in
                `[0, num_edge_channels - 1]` for each active tree vertex (`-1` = marginalize over slots).
            clamped_mask: Optional 1D `uint8` / boolean mask of clamped vertices.
            lr_scale: Cosine/schedule multiplier on `self.learning_rate`.

        Returns:
            float: Total scalar loss for this step.
        """
        rel_csr = self._coerce_csr(csr)
        x_full, c_mask = self._prepare_features(x, rel_csr, clamped_mask)
        n_total = rel_csr.n_nodes

        # 1. Forward pass with layer caches
        h0 = np.ascontiguousarray(x_full @ self.w_in_ + self.b_in_[None, :], dtype=np.float32)
        h_inputs: list[np.ndarray] = []
        pre_norms: list[np.ndarray] = []
        rms_invs: list[np.ndarray] = []

        h_curr = h0
        for lay in range(self.num_layers):
            h_inputs.append(h_curr)
            h_curr, pn, ri = _mpnn_layer_forward(
                h_curr,
                rel_csr.indptr,
                rel_csr.src_indices,
                rel_csr.rel_types,
                rel_csr.edge_weights,
                self.w_rel_[lay],
                self.b_rel_[lay],
                c_mask,
                h0,
            )
            pre_norms.append(pn)
            rms_invs.append(ri)

        total_loss = 0.0
        d_h_final = np.zeros_like(h_curr)

        # 2a. Node classification cross-entropy loss & gradient
        d_w_node = np.zeros_like(self.w_node_)
        d_b_node = np.zeros_like(self.b_node_)
        if node_targets is not None:
            y_node = np.asarray(node_targets, dtype=np.int32)
            n_sup = min(len(y_node), n_total)
            w_n = (
                np.asarray(node_weights, dtype=np.float32)
                if node_weights is not None
                else np.ones(n_sup, dtype=np.float32)
            )
            valid_n = np.where((y_node[:n_sup] >= 0) & (y_node[:n_sup] < self.num_node_classes))[0]
            if len(valid_n) > 0:
                logits_n = h_curr[valid_n] @ self.w_node_ + self.b_node_[None, :]
                max_l = np.max(logits_n, axis=1, keepdims=True)
                exp_l = np.exp(logits_n - max_l)
                probs_n = exp_l / np.maximum(np.sum(exp_l, axis=1, keepdims=True), np.float32(1e-12))

                targets_v = y_node[valid_n]
                weights_v = w_n[valid_n] / np.float32(max(len(valid_n), 1))
                picked_p = probs_n[np.arange(len(valid_n)), targets_v]
                total_loss += float(np.sum(-weights_v * np.log(picked_p + np.float32(1e-10))))

                d_logits_n = probs_n.copy()
                d_logits_n[np.arange(len(valid_n)), targets_v] -= np.float32(1.0)
                d_logits_n *= weights_v[:, None]

                d_w_node = h_curr[valid_n].T @ d_logits_n
                d_b_node = np.sum(d_logits_n, axis=0)
                d_h_final[valid_n] += d_logits_n @ self.w_node_.T

        # 2b. Directed arborescence parent+slot cross-entropy loss & gradient
        d_w_esrc = np.zeros_like(self.w_edge_src_)
        d_w_edst = np.zeros_like(self.w_edge_dst_)
        d_w_echan = np.zeros_like(self.w_edge_chan_)
        d_b_echan = np.zeros_like(self.b_edge_chan_)

        if edge_parent_targets is not None:
            e_idx = (
                np.asarray(edge_node_indices, dtype=np.int32)
                if edge_node_indices is not None
                else np.arange(len(edge_parent_targets), dtype=np.int32)
            )
            p_targets = np.asarray(edge_parent_targets, dtype=np.int32)
            s_targets = (
                np.asarray(edge_slot_targets, dtype=np.int32)
                if edge_slot_targets is not None
                else np.zeros(len(p_targets), dtype=np.int32)
            )
            m_sub = len(e_idx)
            if m_sub >= 2:
                h_sub = np.ascontiguousarray(h_curr[e_idx], dtype=np.float32)
                u_proj = np.ascontiguousarray(h_sub @ self.w_edge_src_, dtype=np.float32)
                v_proj = np.ascontiguousarray(h_sub @ self.w_edge_dst_, dtype=np.float32)
                scores = _bilinear_edge_scores_forward(u_proj, v_proj, self.w_edge_chan_, self.b_edge_chan_)
                d_scores = np.zeros_like(scores)
                sup_children = np.where((p_targets >= 0) & (p_targets < m_sub))[0]
                if len(sup_children) > 0:
                    inv_m = np.float32(1.0 / len(sup_children))
                    for v_loc in sup_children:
                        u_true = int(p_targets[v_loc])
                        if u_true == v_loc:
                            continue
                        s_true = int(s_targets[v_loc])
                        if s_true < 0 or s_true >= self.num_edge_channels:
                            s_true = 0
                        col_logits = scores[:, v_loc, :].reshape(-1)
                        max_c = float(np.max(col_logits))
                        exp_c = np.exp(col_logits - max_c)
                        sum_c = float(np.sum(exp_c)) + 1e-12
                        probs_c = (exp_c / sum_c).reshape(m_sub, self.num_edge_channels)

                        p_gold = float(probs_c[u_true, s_true])
                        total_loss += float(-inv_m * np.log(max(p_gold, 1e-10)))

                        grad_slice = probs_c * inv_m
                        grad_slice[u_true, s_true] -= inv_m
                        grad_slice[v_loc, :] = 0.0
                        d_scores[:, v_loc, :] += grad_slice

                    d_u_proj, d_v_proj, d_w_echan, d_b_echan = _bilinear_edge_scores_backward(
                        d_scores,
                        u_proj,
                        v_proj,
                        self.w_edge_chan_,
                    )
                    d_w_esrc = h_sub.T @ d_u_proj
                    d_w_edst = h_sub.T @ d_v_proj
                    d_h_final[e_idx] += d_u_proj @ self.w_edge_src_.T + d_v_proj @ self.w_edge_dst_.T

        # 3. Backward pass through L relational MPNN layers
        d_w_rel_all = np.zeros_like(self.w_rel_)
        d_b_rel_all = np.zeros_like(self.b_rel_)
        d_h_curr = d_h_final
        d_h0_accum = np.zeros_like(h0)
        clamped_idx = np.flatnonzero(c_mask)
        inv_d = np.float32(1.0 / max(self.hidden_dim, 1))

        for lay in range(self.num_layers - 1, -1, -1):
            if len(clamped_idx) > 0:
                ir = rms_invs[lay][clamped_idx, None]
                pn_c = pre_norms[lay][clamped_idx]
                dh_c = d_h_curr[clamped_idx]
                dot_val = np.sum(dh_c * pn_c, axis=1, keepdims=True)
                d_h0_accum[clamped_idx] += np.float32(0.5) * (dh_c * ir - pn_c * (ir * ir * ir * inv_d) * dot_val)

            d_h_curr, dw_l, db_l = _mpnn_layer_backward(
                d_h_curr,
                h_inputs[lay],
                pre_norms[lay],
                rms_invs[lay],
                rel_csr.indptr,
                rel_csr.src_indices,
                rel_csr.rel_types,
                rel_csr.edge_weights,
                self.w_rel_[lay],
                c_mask,
            )
            d_w_rel_all[lay] = dw_l
            d_b_rel_all[lay] = db_l

        d_h0_total = d_h_curr + d_h0_accum
        d_w_in = x_full.T @ d_h0_total
        d_b_in = np.sum(d_h0_total, axis=0)

        # 4. AdamW parameter update
        self._step += 1
        bc1 = 1.0 - (0.9**self._step)
        bc2 = 1.0 - (0.999**self._step)
        lr_t = float(self.learning_rate * lr_scale)
        step_size = float(lr_t * np.sqrt(bc2) / bc1)
        wd_factor = float(1.0 - lr_t * self.weight_decay)

        _adamw_step(self.w_in_, d_w_in, self._m_w_in, self._v_w_in, step_size, wd_factor, True)
        _adamw_step(self.b_in_, d_b_in, self._m_b_in, self._v_b_in, step_size, wd_factor, False)
        _adamw_step(self.w_rel_, d_w_rel_all, self._m_w_rel, self._v_w_rel, step_size, wd_factor, True)
        _adamw_step(self.b_rel_, d_b_rel_all, self._m_b_rel, self._v_b_rel, step_size, wd_factor, False)

        if node_targets is not None:
            _adamw_step(self.w_node_, d_w_node, self._m_w_node, self._v_w_node, step_size, wd_factor, True)
            _adamw_step(self.b_node_, d_b_node, self._m_b_node, self._v_b_node, step_size, wd_factor, False)

        if edge_parent_targets is not None:
            _adamw_step(self.w_edge_src_, d_w_esrc, self._m_w_esrc, self._v_w_esrc, step_size, wd_factor, True)
            _adamw_step(self.w_edge_dst_, d_w_edst, self._m_w_edst, self._v_w_edst, step_size, wd_factor, True)
            _adamw_step(self.w_edge_chan_, d_w_echan, self._m_w_echan, self._v_w_echan, step_size, wd_factor, True)
            _adamw_step(self.b_edge_chan_, d_b_echan, self._m_b_echan, self._v_b_echan, step_size, wd_factor, False)

        self.loss_history_.append(total_loss)
        return total_loss

    def save(self, path: Path | str) -> Path:
        r"""Save model weights and hyperparameters to a compressed `.npz` archive (`allow_pickle=False`)."""
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = np.array(
            [
                self.in_dim,
                self.hidden_dim,
                self.num_node_classes,
                self.num_edge_channels,
                self.num_relations,
                self.num_layers,
                self.bilinear_rank,
                -1 if self.random_state is None else int(self.random_state),
            ],
            dtype=np.int64,
        )
        float_meta = np.array([self.learning_rate, self.weight_decay], dtype=np.float64)
        np.savez_compressed(
            p,
            meta=meta,
            float_meta=float_meta,
            w_in=self.w_in_,
            b_in=self.b_in_,
            w_rel=self.w_rel_,
            b_rel=self.b_rel_,
            w_node=self.w_node_,
            b_node=self.b_node_,
            w_edge_src=self.w_edge_src_,
            w_edge_dst=self.w_edge_dst_,
            w_edge_chan=self.w_edge_chan_,
            b_edge_chan=self.b_edge_chan_,
            loss_history=np.array(self.loss_history_, dtype=np.float32),
        )
        return p

    @classmethod
    def load(cls, path: Path | str) -> Self:
        r"""Load a [`RelationalMPNN`][seroverse.core.graph.gnn.RelationalMPNN] from a compressed `.npz` archive."""
        p = Path(path)
        if not p.suffix and not p.exists():
            p = p.with_suffix(".npz")
        with np.load(p, allow_pickle=False) as data:
            meta = data["meta"]
            float_meta = data["float_meta"]
            rs = None if int(meta[7]) < 0 else int(meta[7])
            inst = cls(
                in_dim=int(meta[0]),
                hidden_dim=int(meta[1]),
                num_node_classes=int(meta[2]),
                num_edge_channels=int(meta[3]),
                num_relations=int(meta[4]),
                num_layers=int(meta[5]),
                bilinear_rank=int(meta[6]),
                learning_rate=float(float_meta[0]),
                weight_decay=float(float_meta[1]),
                random_state=rs,
            )
            inst.w_in_ = data["w_in"].astype(np.float32)
            inst.b_in_ = data["b_in"].astype(np.float32)
            inst.w_rel_ = data["w_rel"].astype(np.float32)
            inst.b_rel_ = data["b_rel"].astype(np.float32)
            inst.w_node_ = data["w_node"].astype(np.float32)
            inst.b_node_ = data["b_node"].astype(np.float32)
            inst.w_edge_src_ = data["w_edge_src"].astype(np.float32)
            inst.w_edge_dst_ = data["w_edge_dst"].astype(np.float32)
            inst.w_edge_chan_ = data["w_edge_chan"].astype(np.float32)
            inst.b_edge_chan_ = data["b_edge_chan"].astype(np.float32)
            inst.loss_history_ = [float(x) for x in data["loss_history"].tolist()]
            return inst
