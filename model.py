"""
Spatial Heterogeneous Graph Encoder for Multi-omics Data
=========================================================

Node types: Spot (0), Gene (1), Peak (2)
Edge types: Spot-Spot (0, distance-weighted), Spot-Gene (1),
            Spot-Peak (2), Gene-Peak (3)

Training: self-supervised with RNA/ATAC cluster auxiliary heads.
Evaluation: NMI/ARI vs Joint_clusters (E13+P21 only).

v1: Mean aggregation + distance-weighted Spot-Spot.
    Single conv layer for fast CPU iteration.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

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
        # coords: (N, 2)
        proj = 2.0 * np.pi * (coords @ self.B)  # (N, out_dim//2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class DistanceKernel(nn.Module):
    """Learnable Gaussian distance decay: w(d) = exp(-d^2 / (2 * sigma^2))."""

    def __init__(self, init_sigma=1.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(np.log(init_sigma)))

    def forward(self, distances):
        sigma = torch.exp(self.log_sigma)
        return torch.exp(-distances ** 2 / (2 * sigma ** 2 + 1e-8))


# ============================================================
# 2. Heterogeneous Convolution
# ============================================================

class SpatialHeteroConv(nn.Module):
    """
    One layer of heterogeneous message passing.

    Edge type 0 (Spot-Spot): distance-weighted mean aggregation.
    Edge type 1 (Spot-Gene): bidirectional mean aggregation.
    Edge type 2 (Spot-Peak): bidirectional mean aggregation.
    Edge type 3 (Gene-Peak): bidirectional mean aggregation.
    """

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Per-edge-type message projections (source → msg)
        self.msg_ss = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_sg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_gs = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_sp = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_ps = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_gp = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_pg = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Post-aggregation update per node type
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

    def forward(self, h, edge_index_ss, edge_index_sg, edge_index_sp,
                edge_index_gp, dist_ss, node_type, n_spots, n_genes, n_peaks):
        """
        h: (N_total, hidden_dim) all node embeddings
        edge_index_*: (2, E_*) int64 tensors [source, target]
        dist_ss: (E_ss,) precomputed distances for Spot-Spot edges
        node_type: (N_total,) type ids [0,1,2]
        """
        N = h.size(0)
        device = h.device
        agg = torch.zeros(N, self.hidden_dim, device=device)

        # ---- Edge type 0: Spot → Spot (distance-weighted) ----
        if edge_index_ss is not None and edge_index_ss.size(1) > 0:
            src, dst = edge_index_ss[0], edge_index_ss[1]
            msg = self.msg_ss(h[src])                         # (E, d)
            w = torch.exp(-dist_ss ** 2 / (2.0 * 1.0 ** 2))  # distance decay
            msg = msg * w.unsqueeze(-1)
            agg.index_add_(0, dst, msg)

        # ---- Edge type 1: Spot ↔ Gene ----
        if edge_index_sg is not None and edge_index_sg.size(1) > 0:
            src, dst = edge_index_sg[0], edge_index_sg[1]
            # Determine which direction: spot→gene or gene→spot
            src_is_spot = node_type[src] == 0
            src_is_gene = node_type[src] == 1

            # spot → gene
            if src_is_spot.any():
                s, d = src[src_is_spot], dst[src_is_spot]
                agg.index_add_(0, d, self.msg_sg(h[s]))
            # gene → spot
            if src_is_gene.any():
                s, d = src[src_is_gene], dst[src_is_gene]
                agg.index_add_(0, d, self.msg_gs(h[s]))

        # ---- Edge type 2: Spot ↔ Peak ----
        if edge_index_sp is not None and edge_index_sp.size(1) > 0:
            src, dst = edge_index_sp[0], edge_index_sp[1]
            src_is_spot = node_type[src] == 0
            src_is_peak = node_type[src] == 2

            if src_is_spot.any():
                s, d = src[src_is_spot], dst[src_is_spot]
                agg.index_add_(0, d, self.msg_sp(h[s]))
            if src_is_peak.any():
                s, d = src[src_is_peak], dst[src_is_peak]
                agg.index_add_(0, d, self.msg_ps(h[s]))

        # ---- Edge type 3: Gene ↔ Peak ----
        if edge_index_gp is not None and edge_index_gp.size(1) > 0:
            src, dst = edge_index_gp[0], edge_index_gp[1]
            src_is_gene = node_type[src] == 1
            src_is_peak = node_type[src] == 2

            if src_is_gene.any():
                s, d = src[src_is_gene], dst[src_is_gene]
                agg.index_add_(0, d, self.msg_gp(h[s]))
            if src_is_peak.any():
                s, d = src[src_is_peak], dst[src_is_peak]
                agg.index_add_(0, d, self.msg_pg(h[s]))

        # ---- Per-node-type update with residual ----
        h_new = h.clone()

        spot_mask = node_type == 0
        gene_mask = node_type == 1
        peak_mask = node_type == 2

        if spot_mask.any():
            spot_agg = agg[spot_mask]
            spot_h = h[spot_mask]
            deg = self._compute_degree(spot_mask, [edge_index_ss, edge_index_sg, edge_index_sp])
            spot_agg = spot_agg / (deg.unsqueeze(-1) + 1e-8)
            h_new[spot_mask] = self.norm_spot(spot_h + self.drop(self.update_spot(spot_agg)))

        if gene_mask.any():
            gene_agg = agg[gene_mask]
            gene_h = h[gene_mask]
            deg = self._compute_degree(gene_mask, [edge_index_sg, edge_index_gp])
            gene_agg = gene_agg / (deg.unsqueeze(-1) + 1e-8)
            h_new[gene_mask] = self.norm_gene(gene_h + self.drop(self.update_gene(gene_agg)))

        if peak_mask.any():
            peak_agg = agg[peak_mask]
            peak_h = h[peak_mask]
            deg = self._compute_degree(peak_mask, [edge_index_sp, edge_index_gp])
            peak_agg = peak_agg / (deg.unsqueeze(-1) + 1e-8)
            h_new[peak_mask] = self.norm_peak(peak_h + self.drop(self.update_peak(peak_agg)))

        return h_new

    def _compute_degree(self, mask, edge_list):
        """Compute in-degree for nodes in mask across given edge lists."""
        deg = torch.zeros(mask.sum(), device=mask.device)
        offset = mask.nonzero(as_tuple=True)[0].min()
        for edges in edge_list:
            if edges is not None and edges.size(1) > 0:
                local_dst = edges[1] - offset
                valid = (local_dst >= 0) & (local_dst < deg.size(0))
                if valid.any():
                    deg.index_add_(0, local_dst[valid], torch.ones(valid.sum(), device=deg.device))
        return deg


# ============================================================
# 3. Main Encoder
# ============================================================

class SpatialHeteroEncoder(nn.Module):
    """
    Full encoder: feature projection → conv layers → embeddings.
    """

    def __init__(self, n_spots, n_genes, n_peaks,
                 rna_dim, atac_dim, gene_in_dim=None, peak_in_dim=None,
                 hidden_dim=256, pos_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        if gene_in_dim is None:
            gene_in_dim = n_spots
        if peak_in_dim is None:
            peak_in_dim = n_spots

        self.hidden_dim = hidden_dim
        self.n_spots = n_spots
        self.n_genes = n_genes
        self.n_peaks = n_peaks

        # Position encoding for spot coordinates
        self.pos_enc = FourierPositionEncoding(coord_dim=2, out_dim=pos_dim)

        # --- Spot feature projections (split RNA / ATAC) ---
        self.spot_rna_proj = nn.Sequential(
            nn.Linear(rna_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.spot_atac_proj = nn.Sequential(
            nn.Linear(atac_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.spot_fuse = nn.Linear(128 + 128 + pos_dim, hidden_dim)

        # --- Gene / Peak feature projections ---
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.peak_proj = nn.Sequential(
            nn.Linear(peak_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Graph convolution layers ---
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SpatialHeteroConv(hidden_dim, dropout))

        # --- RNA/ATAC view projectors (for cross-modal alignment loss) ---
        self.rna_view_proj = nn.Linear(hidden_dim, 128)
        self.atac_view_proj = nn.Linear(hidden_dim, 128)

    def get_initial_embeddings(self, spot_rna, spot_atac, coords,
                                gene_feat, peak_feat,
                                use_sparse=False, batch_size=2000):
        """
        Compute initial node embeddings.

        Args:
            spot_rna: (n_spots, rna_dim) dense ndarray or sparse CSR
            spot_atac: (n_spots, atac_dim) dense ndarray or sparse CSR
            coords: (n_spots, 2) ndarray
            gene_feat: (n_genes, gene_in_dim) dense ndarray or sparse CSR
            peak_feat: (n_peaks, peak_in_dim) dense ndarray or sparse CSR
            use_sparse: if True, use batched sparse-to-dense conversion
        Returns:
            h: (N_total, hidden_dim)
        """
        device = next(self.parameters()).device

        def to_tensor(x):
            """Convert ndarray or sparse to torch tensor."""
            if sp.issparse(x):
                return torch.tensor(x.toarray(), dtype=torch.float32, device=device)
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=torch.float32)
            return torch.tensor(np.asarray(x), dtype=torch.float32, device=device)

        coords_t = to_tensor(coords)

        # --- Spot embeddings ---
        n_spots = spot_rna.shape[0]
        h_spot = torch.zeros(n_spots, self.hidden_dim, device=device)
        h_pos = self.pos_enc(coords_t)

        if use_sparse:
            # Batched sparse loading
            for start in range(0, n_spots, batch_size):
                end = min(start + batch_size, n_spots)
                rna_batch = to_tensor(spot_rna[start:end])
                atac_batch = to_tensor(spot_atac[start:end])
                h_rna = self.spot_rna_proj(rna_batch)
                h_atac = self.spot_atac_proj(atac_batch)
                h_spot[start:end] = self.spot_fuse(
                    torch.cat([h_rna, h_atac, h_pos[start:end]], dim=-1))
        else:
            # Dense features (SVD) - can process all at once or in batches
            spot_rna_t = to_tensor(spot_rna)
            spot_atac_t = to_tensor(spot_atac)
            for start in range(0, n_spots, batch_size):
                end = min(start + batch_size, n_spots)
                h_rna = self.spot_rna_proj(spot_rna_t[start:end])
                h_atac = self.spot_atac_proj(spot_atac_t[start:end])
                h_spot[start:end] = self.spot_fuse(
                    torch.cat([h_rna, h_atac, h_pos[start:end]], dim=-1))

        # --- Gene embeddings ---
        n_genes = gene_feat.shape[0]
        h_gene = torch.zeros(n_genes, self.hidden_dim, device=device)
        gene_feat_t = to_tensor(gene_feat)
        for start in range(0, n_genes, batch_size):
            end = min(start + batch_size, n_genes)
            h_gene[start:end] = self.gene_proj(gene_feat_t[start:end])

        # --- Peak embeddings ---
        n_peaks = peak_feat.shape[0]
        h_peak = torch.zeros(n_peaks, self.hidden_dim, device=device)
        peak_feat_t = to_tensor(peak_feat)
        for start in range(0, n_peaks, batch_size):
            end = min(start + batch_size, n_peaks)
            h_peak[start:end] = self.peak_proj(peak_feat_t[start:end])

        return torch.cat([h_spot, h_gene, h_peak], dim=0)

    def forward(self, h, edge_index_ss, edge_index_sg, edge_index_sp,
                edge_index_gp, dist_ss, node_type):
        """Run through conv layers, return final embeddings."""
        for conv in self.convs:
            h = conv(h, edge_index_ss, edge_index_sg, edge_index_sp,
                     edge_index_gp, dist_ss, node_type,
                     self.n_spots, self.n_genes, self.n_peaks)
        return h

    def get_views(self, h, edge_index_sg, edge_index_sp, node_type):
        """Extract RNA-view and ATAC-view of spot embeddings for alignment loss.

        RNA-view: spot embedding after aggregating from gene neighbors only.
        ATAC-view: spot embedding after aggregating from peak neighbors only.
        """
        spot_mask = node_type == 0
        spot_h = h[spot_mask]
        n_spots_local = spot_h.size(0)

        # RNA view: mean of connected gene embeddings
        rna_agg = torch.zeros_like(spot_h)
        if edge_index_sg is not None and edge_index_sg.size(1) > 0:
            src, dst = edge_index_sg[0], edge_index_sg[1]
            gene_to_spot = node_type[src] == 1  # gene → spot
            if gene_to_spot.any():
                g_src = src[gene_to_spot]
                s_dst = dst[gene_to_spot]
                gene_msgs = h[g_src]  # global embedding of gene nodes
                rna_agg.index_add_(0, s_dst, gene_msgs)
                deg = torch.zeros(n_spots_local, device=h.device)
                deg.index_add_(0, s_dst, torch.ones(s_dst.size(0), device=h.device))
                valid = deg > 0
                rna_agg[valid] = rna_agg[valid] / deg[valid].unsqueeze(-1)
        rna_view = self.rna_view_proj(spot_h + rna_agg)

        # ATAC view: mean of connected peak embeddings
        atac_agg = torch.zeros_like(spot_h)
        if edge_index_sp is not None and edge_index_sp.size(1) > 0:
            src, dst = edge_index_sp[0], edge_index_sp[1]
            peak_to_spot = node_type[src] == 2  # peak → spot
            if peak_to_spot.any():
                p_src = src[peak_to_spot]
                s_dst = dst[peak_to_spot]
                peak_msgs = h[p_src]
                atac_agg.index_add_(0, s_dst, peak_msgs)
                deg = torch.zeros(n_spots_local, device=h.device)
                deg.index_add_(0, s_dst, torch.ones(s_dst.size(0), device=h.device))
                valid = deg > 0
                atac_agg[valid] = atac_agg[valid] / deg[valid].unsqueeze(-1)
        atac_view = self.atac_view_proj(spot_h + atac_agg)

        return rna_view, atac_view


# ============================================================
# 4. Cluster Heads (per time point × modality)
# ============================================================

class ClusterHeads(nn.Module):
    """
    Separate classification heads for each time point and modality.

    Index mapping:
      rna_heads / atac_heads: dict[time_idx] -> nn.Linear(hidden_dim, n_classes)
    """

    def __init__(self, hidden_dim, n_rna_classes, n_atac_classes):
        """
        Args:
            n_rna_classes: list of 3 ints [E13_n, P21_n, P22_n]
            n_atac_classes: list of 3 ints [E13_n, P21_n, P22_n]
        """
        super().__init__()
        self.rna_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n) if n > 0 else nn.Identity()
            for n in n_rna_classes
        ])
        self.atac_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n) if n > 0 else nn.Identity()
            for n in n_atac_classes
        ])

    def forward(self, spot_emb, time_labels):
        """
        Args:
            spot_emb: (n_spots, hidden_dim)
            time_labels: (n_spots,) int [0, 1, 2]
        Returns:
            rna_logits: list of 3 tensors (or None for empty heads)
            atac_logits: list of 3 tensors
        """
        rna_logits = []
        atac_logits = []
        for t in range(3):
            mask = time_labels == t
            if mask.any() and isinstance(self.rna_heads[t], nn.Linear):
                rna_logits.append(self.rna_heads[t](spot_emb[mask]))
            else:
                rna_logits.append(None)
            if mask.any() and isinstance(self.atac_heads[t], nn.Linear):
                atac_logits.append(self.atac_heads[t](spot_emb[mask]))
            else:
                atac_logits.append(None)
        return rna_logits, atac_logits


# ============================================================
# 5. Loss Functions
# ============================================================

def recon_loss(h_spot, h_gene, h_peak,
               edge_index_sg, edge_index_sp,
               rna_matrix, atac_matrix,
               node_type, n_samples=20000):
    """
    Reconstruction loss: dot product of embeddings should match expression.
    Uses random edge sampling for efficiency.

    Global ID layout:
      Spot: [0, n_spots)
      Gene: [n_spots, n_spots+n_genes)  → local = global - n_spots
      Peak: [n_spots+n_genes, ...)       → local = global - n_spots - n_genes
    """
    device = h_spot.device
    n_spots = h_spot.size(0)
    n_genes = h_gene.size(0)

    # Spot-Gene reconstruction (sample edges)
    loss_sg = torch.tensor(0.0, device=device)
    if edge_index_sg is not None and edge_index_sg.size(1) > 0:
        src, dst = edge_index_sg[0], edge_index_sg[1]
        # gene → spot edges (gene=source, spot=target)
        g2s_mask = node_type[src] == 1
        if g2s_mask.any():
            g2s_idx = g2s_mask.nonzero(as_tuple=True)[0]
            if len(g2s_idx) > n_samples:
                g2s_idx = g2s_idx[torch.randperm(len(g2s_idx), device=device)[:n_samples]]
            gene_global = src[g2s_idx]
            spot_global = dst[g2s_idx]
            gene_local = gene_global - n_spots
            spot_local = spot_global

            pred = (h_gene[gene_local] * h_spot[spot_local]).sum(dim=-1)

            gene_idx_np = gene_local.cpu().numpy()
            spot_idx_np = spot_local.cpu().numpy()
            true_vals = torch.tensor(
                [rna_matrix[g, s] for g, s in zip(gene_idx_np, spot_idx_np)],
                dtype=torch.float32, device=device)
            loss_sg = F.mse_loss(pred, true_vals)

    # Spot-Peak reconstruction (sample edges)
    loss_sp = torch.tensor(0.0, device=device)
    if edge_index_sp is not None and edge_index_sp.size(1) > 0:
        src, dst = edge_index_sp[0], edge_index_sp[1]
        p2s_mask = node_type[src] == 2
        if p2s_mask.any():
            p2s_idx = p2s_mask.nonzero(as_tuple=True)[0]
            if len(p2s_idx) > n_samples:
                p2s_idx = p2s_idx[torch.randperm(len(p2s_idx), device=device)[:n_samples]]
            peak_global = src[p2s_idx]
            spot_global = dst[p2s_idx]
            peak_local = peak_global - n_spots - n_genes
            spot_local = spot_global

            pred = (h_peak[peak_local] * h_spot[spot_local]).sum(dim=-1)

            peak_idx_np = peak_local.cpu().numpy()
            spot_idx_np = spot_local.cpu().numpy()
            true_vals = torch.tensor(
                [atac_matrix[p, s] for p, s in zip(peak_idx_np, spot_idx_np)],
                dtype=torch.float32, device=device)
            loss_sp = F.mse_loss(pred, true_vals)

    return loss_sg + loss_sp


def align_loss(rna_view, atac_view, temperature=0.07):
    """
    Cross-modal InfoNCE: each spot's RNA view should be close to its ATAC view.
    """
    # Normalize
    rna = F.normalize(rna_view, dim=-1)
    atac = F.normalize(atac_view, dim=-1)

    # Similarity matrix
    sim = rna @ atac.T / temperature  # (N, N)

    # Labels: diagonal = positive pairs
    labels = torch.arange(sim.size(0), device=sim.device)

    # Symmetric NCE
    loss_rna = F.cross_entropy(sim, labels)
    loss_atac = F.cross_entropy(sim.T, labels)
    return (loss_rna + loss_atac) / 2


def spatial_loss(h_spot, edge_index_ss, dist_ss):
    """
    Spatial smoothness: nearby spots should have similar embeddings.
    w_ij = exp(-d_ij) → weight for each KNN edge.
    """
    if edge_index_ss is None or edge_index_ss.size(1) == 0:
        return torch.tensor(0.0, device=h_spot.device)

    src, dst = edge_index_ss[0], edge_index_ss[1]
    h_src = h_spot[src]
    h_dst = h_spot[dst]
    diff = (h_src - h_dst).pow(2).sum(dim=-1)
    w = torch.exp(-dist_ss)
    return (w * diff).mean()


def gp_edge_loss(h_gene, h_peak, edge_index_gp, node_type, n_neg=5, n_spots=0):
    """
    Gene-Peak edge prediction: BCE on existing edges + negative sampling.

    edge_index_gp uses global node IDs. We convert to local gene/peak indices.
    gene global ID = n_spots + gene_local
    peak global ID = n_spots + n_genes + peak_local
    """
    if edge_index_gp is None or edge_index_gp.size(1) == 0:
        return torch.tensor(0.0, device=h_gene.device)

    device = h_gene.device
    n_gene = h_gene.size(0)
    n_peak = h_peak.size(0)

    src, dst = edge_index_gp[0], edge_index_gp[1]

    # Convert to local indices
    # Only process edges where source is gene, target is peak
    gene_src = node_type[src] == 1
    peak_src = node_type[src] == 2

    all_scores = []
    all_targets = []

    if gene_src.any():
        g_local = src[gene_src] - n_spots
        p_local = dst[gene_src] - n_spots - n_gene
        pos_score = (h_gene[g_local] * h_peak[p_local]).sum(dim=-1)
        all_scores.append(pos_score)
        all_targets.append(torch.ones(len(pos_score), device=device))

    if peak_src.any():
        p_local = src[peak_src] - n_spots - n_gene
        g_local = dst[peak_src] - n_spots
        pos_score = (h_gene[g_local] * h_peak[p_local]).sum(dim=-1)
        all_scores.append(pos_score)
        all_targets.append(torch.ones(len(pos_score), device=device))

    if len(all_scores) == 0:
        return torch.tensor(0.0, device=device)

    pos_scores = torch.cat(all_scores)
    pos_targets = torch.cat(all_targets)

    # Negative sampling
    n_pos = pos_scores.size(0)
    neg_g = torch.randint(0, n_gene, (n_pos * n_neg,), device=device)
    neg_p = torch.randint(0, n_peak, (n_pos * n_neg,), device=device)
    neg_scores = (h_gene[neg_g] * h_peak[neg_p]).sum(dim=-1)
    neg_targets = torch.zeros_like(neg_scores)

    scores = torch.cat([pos_scores, neg_scores])
    targets = torch.cat([pos_targets, neg_targets])

    return F.binary_cross_entropy_with_logits(scores, targets)


# ============================================================
# 6. Data Loading
# ============================================================

def load_combined_data(data_dir, device='cpu', use_svd=True, svd_dim=100):
    """
    Load the unified graph and prepare tensors for training.

    Args:
        data_dir: path to graph_combined/
        device: 'cpu' or 'cuda'
        use_svd: if True, use TruncatedSVD to reduce feature dimensions
        svd_dim: number of SVD components for each modality
    """
    from sklearn.decomposition import TruncatedSVD

    print("Loading graph data...")

    # Load graph structure
    g = np.load(os.path.join(data_dir, 'graph.npz'), allow_pickle=True)
    n_spots = int(g['n_spots'])
    n_genes = int(g['n_genes'])
    n_peaks = int(g['n_peaks'])
    spot_time_labels = np.load(os.path.join(data_dir, 'spot_time_labels.npy'))
    coords = np.load(os.path.join(data_dir, 'coords.npy'))

    # Combine edges and assign edge types
    edge_parts = []
    edge_types_parts = []

    for etype, key in enumerate(['spot_spot_edges', 'spot_gene_edges',
                                  'spot_peak_edges', 'gene_peak_edges']):
        edges = g[key]
        edge_parts.append(edges)
        edge_types_parts.append(np.full(edges.shape[1], etype, dtype=np.int64))

    all_edges = np.concatenate(edge_parts, axis=1)
    all_edge_types = np.concatenate(edge_types_parts)

    # Split by edge type
    edge_index_ss = torch.tensor(g['spot_spot_edges'], dtype=torch.long, device=device)
    edge_index_sg = torch.tensor(g['spot_gene_edges'], dtype=torch.long, device=device)
    edge_index_sp = torch.tensor(g['spot_peak_edges'], dtype=torch.long, device=device)
    edge_index_gp = torch.tensor(g['gene_peak_edges'], dtype=torch.long, device=device)

    # Precompute Spot-Spot distances
    ss_src = g['spot_spot_edges'][0]
    ss_dst = g['spot_spot_edges'][1]
    ss_dist = np.sqrt(((coords[ss_src] - coords[ss_dst]) ** 2).sum(axis=1))
    dist_ss = torch.tensor(ss_dist, dtype=torch.float32, device=device)

    # Node types
    node_type = np.zeros(n_spots + n_genes + n_peaks, dtype=np.int64)
    node_type[n_spots:n_spots + n_genes] = 1
    node_type[n_spots + n_genes:] = 2
    node_type_t = torch.tensor(node_type, dtype=torch.long, device=device)

    # Expression matrices (sparse CSR)
    rna_matrix = sp.load_npz(os.path.join(data_dir, 'rna_matrix.npz'))
    atac_matrix = sp.load_npz(os.path.join(data_dir, 'atac_matrix.npz'))

    # --- Optional: SVD dimensionality reduction ---
    if use_svd:
        print(f"  Running TruncatedSVD (dim={svd_dim}) on RNA matrix...")
        svd_rna_spot = TruncatedSVD(n_components=svd_dim, n_iter=5, random_state=42)
        spot_rna_svd = svd_rna_spot.fit_transform(rna_matrix.T)
        print(f"    Spot RNA: {rna_matrix.T.shape} -> {spot_rna_svd.shape}"
              f" (explained var: {svd_rna_spot.explained_variance_ratio_.sum():.3f})")

        print(f"  Running TruncatedSVD (dim={svd_dim}) on ATAC matrix...")
        svd_atac_spot = TruncatedSVD(n_components=svd_dim, n_iter=5, random_state=42)
        spot_atac_svd = svd_atac_spot.fit_transform(atac_matrix.T)
        print(f"    Spot ATAC: {atac_matrix.T.shape} -> {spot_atac_svd.shape}"
              f" (explained var: {svd_atac_spot.explained_variance_ratio_.sum():.3f})")

        print(f"  Running TruncatedSVD (dim={svd_dim}) on gene features...")
        svd_gene = TruncatedSVD(n_components=svd_dim, n_iter=5, random_state=42)
        gene_feat_svd = svd_gene.fit_transform(rna_matrix)
        print(f"    Gene: {rna_matrix.shape} -> {gene_feat_svd.shape}"
              f" (explained var: {svd_gene.explained_variance_ratio_.sum():.3f})")

        print(f"  Running TruncatedSVD (dim={svd_dim}) on peak features...")
        svd_peak = TruncatedSVD(n_components=svd_dim, n_iter=5, random_state=42)
        peak_feat_svd = svd_peak.fit_transform(atac_matrix)
        print(f"    Peak: {atac_matrix.shape} -> {peak_feat_svd.shape}"
              f" (explained var: {svd_peak.explained_variance_ratio_.sum():.3f})")

        # Override feature dimensions for model construction
        spot_rna_feat = spot_rna_svd.astype(np.float32)
        spot_atac_feat = spot_atac_svd.astype(np.float32)
        gene_feat_final = gene_feat_svd.astype(np.float32)
        peak_feat_final = peak_feat_svd.astype(np.float32)
        rna_in_dim = svd_dim
        atac_in_dim = svd_dim
        gene_in_dim = svd_dim
        peak_in_dim = svd_dim
        use_sparse_features = False
    else:
        # Use raw sparse features (memory-intensive for CPU)
        spot_rna_feat = rna_matrix.T.tocsr()
        spot_atac_feat = atac_matrix.T.tocsr()
        gene_feat_final = rna_matrix
        peak_feat_final = atac_matrix
        rna_in_dim = rna_matrix.shape[0]
        atac_in_dim = atac_matrix.shape[0]
        gene_in_dim = rna_matrix.shape[1]
        peak_in_dim = atac_matrix.shape[1]
        use_sparse_features = True

    # Time labels
    time_labels_t = torch.tensor(spot_time_labels, dtype=torch.long, device=device)

    # Coordinates
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    # Metadata with cluster labels
    import pandas as pd
    meta = pd.read_csv(os.path.join(data_dir, 'meta.csv'))

    # Encode cluster labels per time point
    time_names_list = ['E13', 'P21', 'P22']
    rna_labels_all = np.full(n_spots, -1, dtype=np.int64)
    atac_labels_all = np.full(n_spots, -1, dtype=np.int64)
    joint_labels_all = np.full(n_spots, -1, dtype=np.int64)
    n_rna_cls = [0, 0, 0]
    n_atac_cls = [0, 0, 0]

    for t, tn in enumerate(time_names_list):
        mask = meta['time_label'].values == tn

        # RNA clusters
        rna_col = meta.loc[mask, 'RNA_clusters'].values
        unique_rna = sorted(set(rna_col))
        n_rna_cls[t] = len(unique_rna)
        rna_map = {v: i for i, v in enumerate(unique_rna)}
        spot_indices = np.where(spot_time_labels == t)[0]
        for si, val in zip(spot_indices, rna_col):
            rna_labels_all[si] = rna_map[val]

        # ATAC clusters
        atac_col = meta.loc[mask, 'ATAC_clusters'].values
        unique_atac = sorted(set(atac_col))
        n_atac_cls[t] = len(unique_atac)
        atac_map = {v: i for i, v in enumerate(unique_atac)}
        for si, val in zip(spot_indices, atac_col):
            atac_labels_all[si] = atac_map[val]

        # Joint clusters (only for E13, P21)
        if 'Joint_clusters' in meta.columns:
            joint_col = meta.loc[mask, 'Joint_clusters'].dropna()
            if len(joint_col) > 0:
                unique_joint = sorted(set(joint_col.values))
                joint_map = {v: i for i, v in enumerate(unique_joint)}
                joint_indices = np.where(spot_time_labels == t)[0]
                for si, val in zip(joint_indices, joint_col.values):
                    joint_labels_all[si] = joint_map[val]

    rna_labels_t = torch.tensor(rna_labels_all, dtype=torch.long, device=device)
    atac_labels_t = torch.tensor(atac_labels_all, dtype=torch.long, device=device)
    joint_labels_t = torch.tensor(joint_labels_all, dtype=torch.long, device=device)

    # Evaluation mask (spots with valid Joint_clusters)
    eval_mask = joint_labels_all >= 0

    print(f"  Nodes: {n_spots} spots + {n_genes} genes + {n_peaks} peaks = {n_spots+n_genes+n_peaks}")
    print(f"  Edges: SS={edge_index_ss.size(1):,}, SG={edge_index_sg.size(1):,}, "
          f"SP={edge_index_sp.size(1):,}, GP={edge_index_gp.size(1):,}")
    print(f"  RNA classes per time: {n_rna_cls}")
    print(f"  ATAC classes per time: {n_atac_cls}")
    print(f"  Joint eval spots: {eval_mask.sum()} / {n_spots}")
    if use_svd:
        print(f"  Feature mode: SVD-reduced ({svd_dim} dims)")
    else:
        print(f"  Feature mode: raw sparse")

    return {
        'n_spots': n_spots,
        'n_genes': n_genes,
        'n_peaks': n_peaks,
        'rna_dim': rna_in_dim,
        'atac_dim': atac_in_dim,
        'gene_in_dim': gene_in_dim,
        'peak_in_dim': peak_in_dim,
        'n_rna_cls': n_rna_cls,
        'n_atac_cls': n_atac_cls,
        'edge_index_ss': edge_index_ss,
        'edge_index_sg': edge_index_sg,
        'edge_index_sp': edge_index_sp,
        'edge_index_gp': edge_index_gp,
        'dist_ss': dist_ss,
        'node_type': node_type_t,
        'time_labels': time_labels_t,
        'spot_rna': spot_rna_feat,
        'spot_atac': spot_atac_feat,
        'gene_feat': gene_feat_final,
        'peak_feat': peak_feat_final,
        'coords': coords_t,
        'rna_labels': rna_labels_t,
        'atac_labels': atac_labels_t,
        'joint_labels': joint_labels_t,
        'eval_mask': eval_mask,
        'rna_matrix': rna_matrix,
        'atac_matrix': atac_matrix,
        'spot_time_labels': spot_time_labels,
        'use_sparse_features': use_sparse_features,
    }


# ============================================================
# 7. Trainer
# ============================================================

class Trainer:
    def __init__(self, model, data, device='cpu',
                 lr=1e-3, wd=1e-4,
                 lambda_align=0.1, lambda_spatial=0.1,
                 lambda_gp=1.0, lambda_rna=0.5, lambda_atac=0.5,
                 temperature=0.07):
        self.model = model.to(device)
        self.data = data
        self.device = device

        # Loss weights
        self.lambda_align = lambda_align
        self.lambda_spatial = lambda_spatial
        self.lambda_gp = lambda_gp
        self.lambda_rna = lambda_rna
        self.lambda_atac = lambda_atac
        self.temperature = temperature

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.5, patience=10, verbose=True)

        # Cluster heads
        self.cluster_heads = ClusterHeads(
            model.hidden_dim,
            data['n_rna_cls'],
            data['n_atac_cls'],
        ).to(device)
        self.cluster_optimizer = torch.optim.AdamW(
            self.cluster_heads.parameters(), lr=lr, weight_decay=wd)

        self.best_nmi = -1
        self.best_ari = -1

    def train_step(self, h0):
        """Single training step, returns loss dict."""
        model = self.model
        data = self.data
        device = self.device

        # Forward
        h = model(
            h0,
            data['edge_index_ss'], data['edge_index_sg'],
            data['edge_index_sp'], data['edge_index_gp'],
            data['dist_ss'], data['node_type'],
        )

        # Split embeddings
        h_spot = h[:model.n_spots]
        h_gene = h[model.n_spots:model.n_spots + model.n_genes]
        h_peak = h[model.n_spots + model.n_genes:]

        # --- Loss 1: Reconstruction ---
        loss_recon = recon_loss(
            h_spot, h_gene, h_peak,
            data['edge_index_sg'], data['edge_index_sp'],
            data['rna_matrix'], data['atac_matrix'],
            data['node_type'],
        )

        # --- Loss 2: Cross-modal alignment ---
        rna_view, atac_view = model.get_views(
            h, data['edge_index_sg'], data['edge_index_sp'], data['node_type'])
        loss_align = align_loss(rna_view, atac_view, self.temperature)

        # --- Loss 3: Spatial smoothness ---
        loss_spatial = spatial_loss(h_spot, data['edge_index_ss'], data['dist_ss'])

        # --- Loss 4: Gene-Peak edge prediction ---
        loss_gp = gp_edge_loss(h_gene, h_peak, data['edge_index_gp'],
                               data['node_type'], n_spots=model.n_spots)

        # --- Loss 5+6: Cluster prediction (per time point) ---
        rna_logits, atac_logits = self.cluster_heads(h_spot, data['time_labels'])

        loss_rna = torch.tensor(0.0, device=device)
        loss_atac = torch.tensor(0.0, device=device)
        rna_labels = data['rna_labels']
        atac_labels = data['atac_labels']
        time_labels = data['time_labels']

        for t in range(3):
            mask = time_labels == t
            if mask.any():
                if rna_logits[t] is not None:
                    valid = rna_labels[mask] >= 0
                    if valid.any():
                        loss_rna += F.cross_entropy(
                            rna_logits[t][valid], rna_labels[mask][valid])
                if atac_logits[t] is not None:
                    valid = atac_labels[mask] >= 0
                    if valid.any():
                        loss_atac += F.cross_entropy(
                            atac_logits[t][valid], atac_labels[mask][valid])

        # --- Total loss ---
        total = (loss_recon +
                 self.lambda_align * loss_align +
                 self.lambda_spatial * loss_spatial +
                 self.lambda_gp * loss_gp +
                 self.lambda_rna * loss_rna +
                 self.lambda_atac * loss_atac)

        return total, {
            'recon': loss_recon.item(),
            'align': loss_align.item(),
            'spatial': loss_spatial.item(),
            'gp': loss_gp.item(),
            'rna': loss_rna.item(),
            'atac': loss_atac.item(),
            'total': total.item(),
        }

    def evaluate(self, h_spot):
        """Compute NMI/ARI vs Joint_clusters on eval spots (E13+P21)."""
        mask = self.data['eval_mask']
        if mask.sum() < 2:
            return {'nmi': 0.0, 'ari': 0.0}

        emb = h_spot[mask].detach().cpu().numpy()
        labels = self.data['joint_labels'][mask].cpu().numpy()
        n_clusters = len(set(labels))

        # KMeans
        if n_clusters >= 2:
            pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(emb)
            nmi = normalized_mutual_info_score(labels, pred)
            ari = adjusted_rand_score(labels, pred)
        else:
            nmi, ari = 0.0, 0.0

        return {'nmi': nmi, 'ari': ari, 'n_clusters': n_clusters}

    def train(self, epochs=100, eval_every=5):
        """Full training loop."""
        data = self.data
        device = self.device

        # Compute initial embeddings once
        print("Computing initial node embeddings...")
        self.model.eval()
        with torch.no_grad():
            h0 = self.model.get_initial_embeddings(
                data['spot_rna'], data['spot_atac'], data['coords'],
                data['gene_feat'], data['peak_feat'],
                use_sparse=data.get('use_sparse_features', False),
            )
        print(f"  h0 shape: {h0.shape}")

        history = []
        for epoch in range(epochs):
            self.model.train()
            self.cluster_heads.train()

            self.optimizer.zero_grad()
            self.cluster_optimizer.zero_grad()

            total_loss, losses = self.train_step(h0)
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(self.cluster_heads.parameters(), 5.0)

            self.optimizer.step()
            self.cluster_optimizer.step()
            self.scheduler.step(total_loss)

            # Evaluation
            metrics = {'epoch': epoch + 1}
            metrics.update(losses)

            if (epoch + 1) % eval_every == 0 or epoch == 0 or epoch == epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    h = self.model(
                        h0,
                        data['edge_index_ss'], data['edge_index_sg'],
                        data['edge_index_sp'], data['edge_index_gp'],
                        data['dist_ss'], data['node_type'],
                    )
                    h_spot = h[:self.model.n_spots]
                    eval_m = self.evaluate(h_spot)
                    metrics.update(eval_m)

                    if eval_m['nmi'] > self.best_nmi:
                        self.best_nmi = eval_m['nmi']
                    if eval_m['ari'] > self.best_ari:
                        self.best_ari = eval_m['ari']

            history.append(metrics)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                nmi_str = f"NMI={metrics.get('nmi', 0):.4f}" if 'nmi' in metrics else ""
                ari_str = f"ARI={metrics.get('ari', 0):.4f}" if 'ari' in metrics else ""
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss {losses['total']:.4f} | "
                      f"R={losses['recon']:.4f} A={losses['align']:.4f} "
                      f"S={losses['spatial']:.4f} GP={losses['gp']:.4f} "
                      f"rna={losses['rna']:.4f} atac={losses['atac']:.4f} "
                      f"{nmi_str} {ari_str}")

        print(f"\nBest: NMI={self.best_nmi:.4f}, ARI={self.best_ari:.4f}")
        return history

    def get_embeddings(self):
        """Extract final embeddings for all nodes."""
        self.model.eval()
        with torch.no_grad():
            h0 = self.model.get_initial_embeddings(
                self.data['spot_rna'], self.data['spot_atac'], self.data['coords'],
                self.data['gene_feat'], self.data['peak_feat'],
                use_sparse=self.data.get('use_sparse_features', False),
            )
            h = self.model(
                h0,
                self.data['edge_index_ss'], self.data['edge_index_sg'],
                self.data['edge_index_sp'], self.data['edge_index_gp'],
                self.data['dist_ss'], self.data['node_type'],
            )
        n_s, n_g = self.model.n_spots, self.model.n_genes
        return {
            'spot': h[:n_s].detach().cpu().numpy(),
            'gene': h[n_s:n_s + n_g].detach().cpu().numpy(),
            'peak': h[n_s + n_g:].detach().cpu().numpy(),
        }

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'cluster_heads': self.cluster_heads.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"Saved to {path}")
