"""
Unsupervised Spatial Heterogeneous Graph Encoder
==================================================

No external labels used in training. Pure self-supervised:
  L = L_recon + λ1·L_align + λ2·L_spatial + λ3·L_gp

GPU-ready with multi-head distance-aware attention.
Evaluation: NMI/ARI vs Joint_clusters (not used in training).
"""

import os, sys, json, warnings
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score,
    adjusted_mutual_info_score, mutual_info_score,
    fowlkes_mallows_score,
)
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Import shared loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.loader import load_combined_data, get_device


# ============================================================
# 1. Utility Components
# ============================================================

class FourierPositionEncoding(nn.Module):
    """Map (x, y) coordinates to high-frequency Fourier features."""

    def __init__(self, coord_dim=2, out_dim=64, sigma=1.0):
        super().__init__()
        B = torch.randn(coord_dim, out_dim // 2) * sigma
        self.register_buffer('B', B)

    def forward(self, coords):
        proj = 2.0 * np.pi * (coords @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class DistanceKernel(nn.Module):
    """Learnable Gaussian distance decay applied to attention weights."""

    def __init__(self, init_sigma=1.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(np.log(init_sigma)))

    def forward(self, distances):
        sigma = torch.exp(self.log_sigma)
        return torch.exp(-distances ** 2 / (2 * sigma ** 2 + 1e-8))


# ============================================================
# 2. Heterogeneous Convolution with Distance-Aware Attention
# ============================================================

class SpatialHeteroConv(nn.Module):
    """
    One layer of heterogeneous message passing.

    Edge type 0 (Spot-Spot): distance-aware multi-head attention.
    Edge type 1 (Spot-Gene): bidirectional multi-head attention.
    Edge type 2 (Spot-Peak): bidirectional multi-head attention.
    Edge type 3 (Gene-Peak): bidirectional multi-head attention.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        # Per-edge-type QKV projections (for target nodes)
        # Spot queries (same for all edge types where spot is target)
        self.q_spot = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_gene = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_peak = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Source key/value projections (per edge-role)
        self.k_spot = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_spot = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_gene = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_gene = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_peak = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_peak = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Distance kernel (only for Spot-Spot)
        self.dist_kernel = DistanceKernel(init_sigma=1.0)

        # Per-node-type update MLPs
        self.update_spot = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.update_gene = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.update_peak = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.norm_spot = nn.LayerNorm(hidden_dim)
        self.norm_gene = nn.LayerNorm(hidden_dim)
        self.norm_peak = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _attn_chunk_pass1(q_mh, k_mh, es, ed, scale, dist_chunk, log_sigma):
        """Pure-tensor chunk forward for pass 1: returns exp(attn) per edge.
        DistanceKernel logic inlined to keep the function self-free."""
        attn = (q_mh[ed] * k_mh[es]).sum(dim=-1) / scale
        if dist_chunk is not None:
            sigma_sq = torch.exp(2 * log_sigma)
            w = torch.exp(-(dist_chunk * dist_chunk) / (2 * sigma_sq + 1e-8))
            attn = attn * w.unsqueeze(-1)
        return attn.clamp(max=20.0).exp()

    @staticmethod
    def _attn_chunk_pass2(q_mh, k_mh, v_mh, es, ed, scale,
                          dist_chunk, log_sigma, attn_max_ed):
        """Pure-tensor chunk forward for pass 2: returns weighted message."""
        attn = (q_mh[ed] * k_mh[es]).sum(dim=-1) / scale
        if dist_chunk is not None:
            sigma_sq = torch.exp(2 * log_sigma)
            w = torch.exp(-(dist_chunk * dist_chunk) / (2 * sigma_sq + 1e-8))
            attn = attn * w.unsqueeze(-1)
        attn_norm = attn.clamp(max=20.0).exp() / attn_max_ed
        return v_mh[es] * attn_norm.unsqueeze(-1)

    def _attn_aggregate(self, q, k, v, edge_src, edge_dst,
                        dist=None, num_nodes=None, chunk_size=100_000):
        """Multi-head attention aggregation, two-pass softmax.

        Each chunk is wrapped in checkpoint() so autograd does not retain
        the intermediate (E_chunk, H, dk) tensors. Chunk functions are
        @staticmethod with pure-tensor signatures so checkpoint does not
        silently fall back."""
        if num_nodes is None:
            num_nodes = q.size(0)

        H, dk = self.num_heads, self.d_k
        scale = dk ** 0.5
        device = q.device

        q_mh = q.view(-1, H, dk)
        k_mh = k.view(-1, H, dk)
        v_mh = v.view(-1, H, dk)

        E = edge_src.size(0)
        use_ckpt = self.training and torch.is_grad_enabled()
        log_sigma = self.dist_kernel.log_sigma  # nn.Parameter shared by all chunks

        # Pass 1: per-dst softmax denominators
        attn_max = torch.zeros(num_nodes, H, device=device, dtype=q.dtype)
        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            es = edge_src[start:end]
            ed = edge_dst[start:end]
            d = dist[start:end] if dist is not None else None
            if use_ckpt:
                exp_attn = torch.utils.checkpoint.checkpoint(
                    self._attn_chunk_pass1,
                    q_mh, k_mh, es, ed, scale, d, log_sigma,
                    use_reentrant=False)
            else:
                exp_attn = self._attn_chunk_pass1(
                    q_mh, k_mh, es, ed, scale, d, log_sigma)
            attn_max.index_add_(0, ed, exp_attn)
        attn_max = attn_max.clamp(min=1e-8)

        # Pass 2: aggregate weighted values
        agg = torch.zeros(num_nodes, H, dk, device=device, dtype=q.dtype)
        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            es = edge_src[start:end]
            ed = edge_dst[start:end]
            d = dist[start:end] if dist is not None else None
            attn_max_ed = attn_max[ed]
            if use_ckpt:
                msg = torch.utils.checkpoint.checkpoint(
                    self._attn_chunk_pass2,
                    q_mh, k_mh, v_mh, es, ed, scale, d, log_sigma, attn_max_ed,
                    use_reentrant=False)
            else:
                msg = self._attn_chunk_pass2(
                    q_mh, k_mh, v_mh, es, ed, scale, d, log_sigma, attn_max_ed)
            agg.index_add_(0, ed, msg)

        return agg.flatten(1)

    def forward(self, h, edge_index_ss, edge_index_sg, edge_index_sp,
                edge_index_gp, dist_ss, node_type,
                n_spots, n_genes, n_peaks):
        device = h.device
        N = h.size(0)

        # --- Extract per-type embeddings ---
        spot_mask = node_type == 0
        gene_mask = node_type == 1
        peak_mask = node_type == 2

        h_spot = h[spot_mask]
        h_gene = h[gene_mask]
        h_peak = h[peak_mask]

        # --- Compute Q, K, V ---
        q_spot = self.q_spot(h_spot)
        q_gene = self.q_gene(h_gene)
        q_peak = self.q_peak(h_peak)
        k_spot = self.k_spot(h_spot)
        v_spot = self.v_spot(h_spot)
        k_gene = self.k_gene(h_gene)
        v_gene = self.v_gene(h_gene)
        k_peak = self.k_peak(h_peak)
        v_peak = self.v_peak(h_peak)

        agg_spot = torch.zeros(n_spots, self.hidden_dim, device=device)
        agg_gene = torch.zeros(n_genes, self.hidden_dim, device=device)
        agg_peak = torch.zeros(n_peaks, self.hidden_dim, device=device)

        # ---- Edge type 0: Spot-Spot (distance-aware) ----
        if edge_index_ss is not None and edge_index_ss.size(1) > 0:
            src, dst = edge_index_ss[0], edge_index_ss[1]
            agg_spot += self._attn_aggregate(
                q_spot, k_spot, v_spot, src, dst, dist=dist_ss, num_nodes=n_spots)

        # ---- Edge type 1: Spot ↔ Gene ----
        if edge_index_sg is not None and edge_index_sg.size(1) > 0:
            src, dst = edge_index_sg[0], edge_index_sg[1]
            # Spot → Gene
            s2g = node_type[src] == 0
            if s2g.any():
                s_local = src[s2g]
                g_local = dst[s2g] - n_spots
                agg_gene += self._attn_aggregate(
                    q_gene, k_spot, v_spot, s_local, g_local, num_nodes=n_genes)
            # Gene → Spot
            g2s = node_type[src] == 1
            if g2s.any():
                g_local = src[g2s] - n_spots
                s_local = dst[g2s]
                agg_spot += self._attn_aggregate(
                    q_spot, k_gene, v_gene, g_local, s_local, num_nodes=n_spots)

        # ---- Edge type 2: Spot ↔ Peak ----
        if edge_index_sp is not None and edge_index_sp.size(1) > 0:
            src, dst = edge_index_sp[0], edge_index_sp[1]
            # Spot → Peak
            s2p = node_type[src] == 0
            if s2p.any():
                s_local = src[s2p]
                p_local = dst[s2p] - n_spots - n_genes
                agg_peak += self._attn_aggregate(
                    q_peak, k_spot, v_spot, s_local, p_local, num_nodes=n_peaks)
            # Peak → Spot
            p2s = node_type[src] == 2
            if p2s.any():
                p_local = src[p2s] - n_spots - n_genes
                s_local = dst[p2s]
                agg_spot += self._attn_aggregate(
                    q_spot, k_peak, v_peak, p_local, s_local, num_nodes=n_spots)

        # ---- Edge type 3: Gene ↔ Peak ----
        if edge_index_gp is not None and edge_index_gp.size(1) > 0:
            src, dst = edge_index_gp[0], edge_index_gp[1]
            # Gene → Peak
            g2p = node_type[src] == 1
            if g2p.any():
                g_local = src[g2p] - n_spots
                p_local = dst[g2p] - n_spots - n_genes
                agg_peak += self._attn_aggregate(
                    q_peak, k_gene, v_gene, g_local, p_local, num_nodes=n_peaks)
            # Peak → Gene
            p2g = node_type[src] == 2
            if p2g.any():
                p_local = src[p2g] - n_spots - n_genes
                g_local = dst[p2g] - n_spots
                agg_gene += self._attn_aggregate(
                    q_gene, k_peak, v_peak, p_local, g_local, num_nodes=n_genes)

        # ---- Per-node-type update with residual ----
        h_new = h.clone()

        h_new[spot_mask] = self.norm_spot(
            h_spot + self.drop(self.update_spot(self.out_proj(agg_spot))))
        h_new[gene_mask] = self.norm_gene(
            h_gene + self.drop(self.update_gene(self.out_proj(agg_gene))))
        h_new[peak_mask] = self.norm_peak(
            h_peak + self.drop(self.update_peak(self.out_proj(agg_peak))))

        return h_new


# ============================================================
# 3. Main Encoder
# ============================================================

class UnsupervisedEncoder(nn.Module):
    """Encoder without any cluster supervision heads."""

    def __init__(self, n_spots, n_genes, n_peaks,
                 rna_dim, atac_dim, gene_in_dim, peak_in_dim,
                 hidden_dim=256, pos_dim=64, num_heads=4,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_spots = n_spots
        self.n_genes = n_genes
        self.n_peaks = n_peaks

        # Position encoding
        self.pos_enc = FourierPositionEncoding(coord_dim=2, out_dim=pos_dim)

        # Spot feature projections
        self.spot_rna_proj = nn.Sequential(
            nn.Linear(rna_dim, 128), nn.ReLU(), nn.Dropout(dropout))
        self.spot_atac_proj = nn.Sequential(
            nn.Linear(atac_dim, 128), nn.ReLU(), nn.Dropout(dropout))
        self.spot_fuse = nn.Linear(128 + 128 + pos_dim, hidden_dim)

        # Gene / Peak projections
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.peak_proj = nn.Sequential(
            nn.Linear(peak_in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        # Graph convolution layers
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                SpatialHeteroConv(hidden_dim, num_heads=num_heads, dropout=dropout))

        # RNA/ATAC view projectors (for cross-modal alignment)
        self.rna_view_proj = nn.Linear(hidden_dim, 128)
        self.atac_view_proj = nn.Linear(hidden_dim, 128)

    def get_initial_embeddings(self, data, batch_size=4000):
        """Compute initial embeddings from pre-loaded features."""
        device = next(self.parameters()).device
        spot_rna = data['spot_rna']
        spot_atac = data['spot_atac']
        coords = data['coords']
        gene_feat = data['gene_feat']
        peak_feat = data['peak_feat']

        # Ensure on device
        if not spot_rna.is_cuda and device.type == 'cuda':
            spot_rna = spot_rna.to(device)
            spot_atac = spot_atac.to(device)
            coords = coords.to(device)
            gene_feat = gene_feat.to(device)
            peak_feat = peak_feat.to(device)

        h_pos = self.pos_enc(coords)

        # Spots
        n_spots = spot_rna.size(0)
        h_spot = torch.zeros(n_spots, self.hidden_dim, device=device)
        for start in range(0, n_spots, batch_size):
            end = min(start + batch_size, n_spots)
            hr = self.spot_rna_proj(spot_rna[start:end])
            ha = self.spot_atac_proj(spot_atac[start:end])
            h_spot[start:end] = self.spot_fuse(
                torch.cat([hr, ha, h_pos[start:end]], dim=-1))

        # Genes
        n_genes = gene_feat.size(0)
        h_gene = torch.zeros(n_genes, self.hidden_dim, device=device)
        for start in range(0, n_genes, batch_size):
            end = min(start + batch_size, n_genes)
            h_gene[start:end] = self.gene_proj(gene_feat[start:end])

        # Peaks
        n_peaks = peak_feat.size(0)
        h_peak = torch.zeros(n_peaks, self.hidden_dim, device=device)
        for start in range(0, n_peaks, batch_size):
            end = min(start + batch_size, n_peaks)
            h_peak[start:end] = self.peak_proj(peak_feat[start:end])

        return torch.cat([h_spot, h_gene, h_peak], dim=0)

    def forward(self, h, edge_index_ss, edge_index_sg, edge_index_sp,
                edge_index_gp, dist_ss, node_type):
        for conv in self.convs:
            h = conv(h, edge_index_ss, edge_index_sg, edge_index_sp,
                     edge_index_gp, dist_ss, node_type,
                     self.n_spots, self.n_genes, self.n_peaks)
        return h

    def get_views(self, h, edge_index_sg, edge_index_sp, node_type):
        """RNA-view and ATAC-view of spot embeddings for alignment loss."""
        spot_mask = node_type == 0
        spot_h = h[spot_mask]
        n_spots = spot_h.size(0)

        # RNA view: aggregate from gene neighbors
        rna_agg = torch.zeros_like(spot_h)
        if edge_index_sg is not None and edge_index_sg.size(1) > 0:
            src, dst = edge_index_sg[0], edge_index_sg[1]
            g2s = node_type[src] == 1
            if g2s.any():
                rna_agg.index_add_(0, dst[g2s], h[src[g2s]])
                deg = torch.zeros(n_spots, device=h.device)
                deg.index_add_(0, dst[g2s], torch.ones(g2s.sum(), device=h.device))
                valid = deg > 0
                rna_agg[valid] = rna_agg[valid] / deg[valid].unsqueeze(-1)
        rna_view = self.rna_view_proj(spot_h + rna_agg)

        # ATAC view: aggregate from peak neighbors
        atac_agg = torch.zeros_like(spot_h)
        if edge_index_sp is not None and edge_index_sp.size(1) > 0:
            src, dst = edge_index_sp[0], edge_index_sp[1]
            p2s = node_type[src] == 2
            if p2s.any():
                atac_agg.index_add_(0, dst[p2s], h[src[p2s]])
                deg = torch.zeros(n_spots, device=h.device)
                deg.index_add_(0, dst[p2s], torch.ones(p2s.sum(), device=h.device))
                valid = deg > 0
                atac_agg[valid] = atac_agg[valid] / deg[valid].unsqueeze(-1)
        atac_view = self.atac_view_proj(spot_h + atac_agg)

        return rna_view, atac_view


# ============================================================
# 4. Loss Functions
# ============================================================

def recon_loss(h_spot, h_gene, h_peak, data, n_samples=30000):
    """Reconstruction: dot product of embeddings should match expression."""
    device = h_spot.device
    n_spots = h_spot.size(0)
    scale = h_spot.size(-1) ** 0.5  # 1/sqrt(d) keeps pred magnitude stable across hidden sizes

    # Spot-Gene reconstruction
    sg_loss = torch.tensor(0.0, device=device)
    sg_total = data['sg_gene_idx'].size(0)
    if sg_total > 0:
        n_s = min(n_samples, sg_total)
        idx = torch.randperm(sg_total, device=device)[:n_s]
        g_local = data['sg_gene_idx'][idx]
        s_local = data['sg_spot_idx'][idx]
        pred = (h_gene[g_local] * h_spot[s_local]).sum(dim=-1) / scale
        true = data['sg_true'][idx].to(device)
        sg_loss = F.mse_loss(pred, true)

    # Spot-Peak reconstruction
    sp_loss = torch.tensor(0.0, device=device)
    sp_total = data['sp_peak_idx'].size(0)
    if sp_total > 0:
        n_s = min(n_samples, sp_total)
        idx = torch.randperm(sp_total, device=device)[:n_s]
        p_local = data['sp_peak_idx'][idx]
        s_local = data['sp_spot_idx'][idx]
        pred = (h_peak[p_local] * h_spot[s_local]).sum(dim=-1) / scale
        true = data['sp_true'][idx].to(device)
        sp_loss = F.mse_loss(pred, true)

    return sg_loss + sp_loss


def align_loss(rna_view, atac_view, temperature=0.07):
    """Cross-modal InfoNCE: same spot's RNA/ATAC views should align."""
    rna = F.normalize(rna_view, dim=-1)
    atac = F.normalize(atac_view, dim=-1)
    sim = rna @ atac.T / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def spatial_loss(h_spot, edge_index_ss, dist_ss):
    """Spatial smoothness: nearby spots should be similar."""
    if edge_index_ss is None or edge_index_ss.size(1) == 0:
        return torch.tensor(0.0, device=h_spot.device)
    src, dst = edge_index_ss[0], edge_index_ss[1]
    diff = (h_spot[src] - h_spot[dst]).pow(2).sum(dim=-1)
    w = torch.exp(-dist_ss)
    return (w * diff).mean()


def gp_edge_loss(h_gene, h_peak, edge_index_gp, node_type, n_spots, n_genes, n_neg=5):
    """Gene-Peak edge prediction: BCE + negative sampling."""
    if edge_index_gp is None or edge_index_gp.size(1) == 0:
        return torch.tensor(0.0, device=h_gene.device)

    device = h_gene.device
    src, dst = edge_index_gp[0], edge_index_gp[1]
    g2p = node_type[src] == 1
    p2g = node_type[src] == 2

    pos_scores = []
    if g2p.any():
        g_local = src[g2p] - n_spots
        p_local = dst[g2p] - n_spots - n_genes
        pos_scores.append((h_gene[g_local] * h_peak[p_local]).sum(dim=-1))
    if p2g.any():
        p_local = src[p2g] - n_spots - n_genes
        g_local = dst[p2g] - n_spots
        pos_scores.append((h_gene[g_local] * h_peak[p_local]).sum(dim=-1))

    if not pos_scores:
        return torch.tensor(0.0, device=device)

    pos_scores = torch.cat(pos_scores)
    n_pos = pos_scores.size(0)

    # Negatives
    neg_g = torch.randint(0, h_gene.size(0), (n_pos * n_neg,), device=device)
    neg_p = torch.randint(0, h_peak.size(0), (n_pos * n_neg,), device=device)
    neg_scores = (h_gene[neg_g] * h_peak[neg_p]).sum(dim=-1)

    scores = torch.cat([pos_scores, neg_scores])
    targets = torch.cat([torch.ones(n_pos, device=device),
                         torch.zeros(n_pos * n_neg, device=device)])
    return F.binary_cross_entropy_with_logits(scores, targets)


# ============================================================
# 5. Trainer
# ============================================================

def _cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = int(max(y_pred.max(), y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return w[row, col].sum() / y_pred.size


def _all_metrics(labels, pred):
    return {
        'nmi': normalized_mutual_info_score(labels, pred),
        'ami': adjusted_mutual_info_score(labels, pred),
        'ari': adjusted_rand_score(labels, pred),
        'fmi': fowlkes_mallows_score(labels, pred),
        'mi':  mutual_info_score(labels, pred),
        'acc': _cluster_acc(labels, pred),
    }


class UnsupervisedTrainer:
    def __init__(self, model, data, device='cuda',
                 lr=1e-3, wd=1e-4,
                 lambda_recon=1.0, lambda_align=0.3,
                 lambda_spatial=0.2, lambda_gp=1.0,
                 temperature=0.07, use_amp=False,
                 joint_cluster=False, cluster_mode='per_time'):
        self.model = model.to(device)
        self.data = data
        self.device = device

        self.lambda_recon = lambda_recon
        self.lambda_align = lambda_align
        self.lambda_spatial = lambda_spatial
        self.lambda_gp = lambda_gp
        self.temperature = temperature
        self.use_amp = use_amp and device.type == 'cuda'
        # cluster_mode in {'per_time', 'joint_all', 'joint_e13p21'}
        # joint_cluster=True (legacy) maps to 'joint_all'.
        if joint_cluster and cluster_mode == 'per_time':
            cluster_mode = 'joint_all'
        self.cluster_mode = cluster_mode
        self.joint_cluster = cluster_mode != 'per_time'
        self.scaler = torch.amp.GradScaler() if self.use_amp else None

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=300)

        self.best_nmi = -1
        self.best_ari = -1

    def train_step(self, h0):
        model = self.model
        data = self.data
        device = self.device

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # Forward
            h = model(h0, data['edge_index_ss'], data['edge_index_sg'],
                      data['edge_index_sp'], data['edge_index_gp'],
                      data['dist_ss'], data['node_type'])

            h_spot = h[:model.n_spots]
            h_gene = h[model.n_spots:model.n_spots + model.n_genes]
            h_peak = h[model.n_spots + model.n_genes:]

            # Losses
            loss_recon = recon_loss(h_spot, h_gene, h_peak, data)
            rna_view, atac_view = model.get_views(
                h, data['edge_index_sg'], data['edge_index_sp'], data['node_type'])
            loss_align = align_loss(rna_view, atac_view, self.temperature)
            loss_spatial = spatial_loss(h_spot, data['edge_index_ss'], data['dist_ss'])
            loss_gp = gp_edge_loss(h_gene, h_peak, data['edge_index_gp'],
                                   data['node_type'], model.n_spots, model.n_genes)

            total = (self.lambda_recon * loss_recon +
                     self.lambda_align * loss_align +
                     self.lambda_spatial * loss_spatial +
                     self.lambda_gp * loss_gp)

        return total, {
            'recon': loss_recon.item(), 'align': loss_align.item(),
            'spatial': loss_spatial.item(), 'gp': loss_gp.item(),
            'total': total.item(),
        }

    def evaluate(self, h_spot):
        """Clustering metrics vs Joint_clusters. P22 has no labels and is
        skipped from metric aggregation.

        cluster_mode controls the KMeans pool:
          per_time     : one KMeans per time point (legacy)
          joint_all    : one KMeans over E13+P21+P22, then slice per time
          joint_e13p21 : one KMeans over only E13+P21 spots, then slice per time
        """
        time_labels = self.data['time_labels'].cpu().numpy()
        joint_labels = self.data['joint_labels'].cpu().numpy()
        emb_all = h_spot.detach().cpu().numpy()
        time_names = ['E13', 'P21', 'P22']
        metric_keys = ['nmi', 'ami', 'ari', 'fmi', 'mi', 'acc']

        out = {}
        agg = {k: [] for k in metric_keys}

        if self.cluster_mode in ('joint_all', 'joint_e13p21'):
            # Per-time joint-label sets are disjoint (each encoded 0..k-1
            # independently in loader.py), so the natural joint k is the
            # sum of unique label counts in E13 + P21.
            n_clusters = 0
            for t in (0, 1):
                tm = (time_labels == t) & (joint_labels >= 0)
                if tm.sum() > 0:
                    n_clusters += len(np.unique(joint_labels[tm]))
            if n_clusters < 2:
                return {k: 0.0 for k in metric_keys}

            if self.cluster_mode == 'joint_all':
                pool_mask = np.ones_like(time_labels, dtype=bool)
            else:  # joint_e13p21
                pool_mask = (time_labels == 0) | (time_labels == 1)
            pool_idx = np.where(pool_mask)[0]
            pred_pool = KMeans(n_clusters=n_clusters, n_init=10,
                               random_state=42).fit_predict(emb_all[pool_idx])
            pred_all = np.full(emb_all.shape[0], -1, dtype=np.int64)
            pred_all[pool_idx] = pred_pool

            for t, tn in enumerate(time_names):
                mask = (time_labels == t) & (joint_labels >= 0) & (pred_all >= 0)
                if mask.sum() < 2:
                    continue
                m = _all_metrics(joint_labels[mask], pred_all[mask])
                for k, v in m.items():
                    out[f'{k}_{tn}'] = v
                    agg[k].append(v)
        else:
            for t, tn in enumerate(time_names):
                mask = (time_labels == t) & (joint_labels >= 0)
                if mask.sum() < 2:
                    continue
                emb = emb_all[mask]
                labels = joint_labels[mask]
                n_clusters = len(np.unique(labels))
                if n_clusters < 2:
                    continue
                pred = KMeans(n_clusters=n_clusters, n_init=10,
                              random_state=42).fit_predict(emb)
                m = _all_metrics(labels, pred)
                for k, v in m.items():
                    out[f'{k}_{tn}'] = v
                    agg[k].append(v)

        for k in metric_keys:
            out[k] = float(np.mean(agg[k])) if agg[k] else 0.0
        return out

    def train(self, epochs=200, eval_every=5):
        data = self.data
        device = self.device

        print("Computing initial node embeddings...")
        self.model.eval()
        with torch.no_grad():
            h0 = self.model.get_initial_embeddings(data)
        print(f"  h0: {h0.shape}, device: {h0.device}")

        history = []
        for epoch in range(epochs):
            self.model.train()

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    total_loss, losses = self.train_step(h0)
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss, losses = self.train_step(h0)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            self.scheduler.step()

            metrics = {'epoch': epoch + 1, **losses}

            if (epoch + 1) % eval_every == 0 or epoch == 0 or epoch == epochs - 1:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                self.model.eval()
                with torch.no_grad():
                    h = self.model(h0, data['edge_index_ss'], data['edge_index_sg'],
                                   data['edge_index_sp'], data['edge_index_gp'],
                                   data['dist_ss'], data['node_type'])
                    h_spot = h[:self.model.n_spots]
                    eval_m = self.evaluate(h_spot)
                    metrics.update(eval_m)
                    if eval_m['nmi'] > self.best_nmi:
                        self.best_nmi = eval_m['nmi']
                    if eval_m['ari'] > self.best_ari:
                        self.best_ari = eval_m['ari']
                del h, h_spot
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            history.append(metrics)

            if (epoch + 1) % 10 == 0 or epoch < 10:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:4d} | Loss {losses['total']:.4f} | "
                      f"R={losses['recon']:.3f} A={losses['align']:.3f} "
                      f"S={losses['spatial']:.3f} GP={losses['gp']:.3f} | "
                      f"lr={lr:.2e}")
                if 'nmi_E13' in metrics or 'nmi_P21' in metrics:
                    for tn in ['E13', 'P21']:
                        if f'nmi_{tn}' in metrics:
                            print(f"  {tn}  | NMI={metrics[f'nmi_{tn}']:.4f} "
                                  f"AMI={metrics[f'ami_{tn}']:.4f} "
                                  f"ARI={metrics[f'ari_{tn}']:.4f} "
                                  f"FMI={metrics[f'fmi_{tn}']:.4f} "
                                  f"MI={metrics[f'mi_{tn}']:.4f} "
                                  f"ACC={metrics[f'acc_{tn}']:.4f}")
                    print(f"  avg  | NMI={metrics['nmi']:.4f} "
                          f"AMI={metrics['ami']:.4f} ARI={metrics['ari']:.4f} "
                          f"FMI={metrics['fmi']:.4f} MI={metrics['mi']:.4f} "
                          f"ACC={metrics['acc']:.4f}")

        print(f"\nBest: NMI={self.best_nmi:.4f}, ARI={self.best_ari:.4f}")
        return history

    def get_embeddings(self):
        self.model.eval()
        with torch.no_grad():
            h0 = self.model.get_initial_embeddings(self.data)
            h = self.model(h0, self.data['edge_index_ss'], self.data['edge_index_sg'],
                           self.data['edge_index_sp'], self.data['edge_index_gp'],
                           self.data['dist_ss'], self.data['node_type'])
        ns, ng = self.model.n_spots, self.model.n_genes
        return {
            'spot': h[:ns].detach().cpu().numpy(),
            'gene': h[ns:ns + ng].detach().cpu().numpy(),
            'peak': h[ns + ng:].detach().cpu().numpy(),
        }

    def compute_gp_scores(self, top_k=100, chunk_size=1024):
        """Compute Gene-Peak association scores (chunked over genes)."""
        self.model.eval()
        with torch.no_grad():
            h0 = self.model.get_initial_embeddings(self.data)
            h = self.model(h0, self.data['edge_index_ss'], self.data['edge_index_sg'],
                           self.data['edge_index_sp'], self.data['edge_index_gp'],
                           self.data['dist_ss'], self.data['node_type'])
        ns, ng = self.model.n_spots, self.model.n_genes
        h_gene = h[ns:ns + ng]
        h_peak = h[ns + ng:]
        n_peaks = h_peak.size(0)
        scores = np.empty((ng, n_peaks), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, ng, chunk_size):
                j = min(i + chunk_size, ng)
                block = torch.sigmoid(h_gene[i:j] @ h_peak.T)
                scores[i:j] = block.cpu().numpy()
        return scores

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, path)
        print(f"Saved to {path}")
