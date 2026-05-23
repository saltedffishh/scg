"""
Supervised Spatial Heterogeneous Graph Encoder
================================================

Uses RNA_clusters and ATAC_clusters as training signals.
  L = L_recon + λ1·L_align + λ2·L_spatial + λ3·L_gp
      + λ4·L_rna_cluster + λ5·L_atac_cluster

GPU-ready with multi-head distance-aware attention.
Evaluation: NMI/ARI vs Joint_clusters (held-out).
"""

import os, sys, json, warnings
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.loader import load_combined_data, get_device


# ============================================================
# 1. Encoder (same architecture as unsupervised)
# ============================================================

class FourierPositionEncoding(nn.Module):
    def __init__(self, coord_dim=2, out_dim=64, sigma=1.0):
        super().__init__()
        B = torch.randn(coord_dim, out_dim // 2) * sigma
        self.register_buffer('B', B)

    def forward(self, coords):
        proj = 2.0 * np.pi * (coords @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class DistanceKernel(nn.Module):
    def __init__(self, init_sigma=1.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(np.log(init_sigma)))

    def forward(self, distances):
        sigma = torch.exp(self.log_sigma)
        return torch.exp(-distances ** 2 / (2 * sigma ** 2 + 1e-8))


class SpatialHeteroConv(nn.Module):
    """Multi-head heterogeneous conv with distance-aware Spot-Spot attention."""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.q_spot = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_gene = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_peak = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_spot = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_spot = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_gene = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_gene = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_peak = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_peak = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dist_kernel = DistanceKernel(init_sigma=1.0)

        self.update_spot = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.update_gene = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.update_peak = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))

        self.norm_spot = nn.LayerNorm(hidden_dim)
        self.norm_gene = nn.LayerNorm(hidden_dim)
        self.norm_peak = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def _attn_aggregate(self, q, k, v, edge_src, edge_dst,
                        dist=None, num_nodes=None):
        if num_nodes is None:
            num_nodes = q.size(0)
        q_mh = q.view(-1, self.num_heads, self.d_k)
        k_mh = k.view(-1, self.num_heads, self.d_k)
        v_mh = v.view(-1, self.num_heads, self.d_k)

        k_e, v_e, q_e = k_mh[edge_src], v_mh[edge_src], q_mh[edge_dst]
        attn = (q_e * k_e).sum(dim=-1) / (self.d_k ** 0.5)

        if dist is not None:
            attn = attn * self.dist_kernel(dist).unsqueeze(-1)

        attn_max = torch.zeros(num_nodes, self.num_heads, device=q.device)
        attn_max = attn_max.index_add(0, edge_dst, attn.exp())
        attn_max = attn_max.clamp(min=1e-8)
        attn = attn / attn_max[edge_dst]

        msg = v_e * attn.unsqueeze(-1)
        agg = torch.zeros(num_nodes, self.num_heads, self.d_k, device=q.device)
        agg = agg.index_add(0, edge_dst, msg)
        return agg.flatten(1)

    def forward(self, h, edge_index_ss, edge_index_sg, edge_index_sp,
                edge_index_gp, dist_ss, node_type,
                n_spots, n_genes, n_peaks):
        device = h.device
        spot_mask, gene_mask, peak_mask = node_type == 0, node_type == 1, node_type == 2
        h_spot, h_gene, h_peak = h[spot_mask], h[gene_mask], h[peak_mask]

        q_spot = self.q_spot(h_spot); q_gene = self.q_gene(h_gene); q_peak = self.q_peak(h_peak)
        k_spot = self.k_spot(h_spot); v_spot = self.v_spot(h_spot)
        k_gene = self.k_gene(h_gene); v_gene = self.v_gene(h_gene)
        k_peak = self.k_peak(h_peak); v_peak = self.v_peak(h_peak)

        agg_spot = torch.zeros(n_spots, self.hidden_dim, device=device)
        agg_gene = torch.zeros(n_genes, self.hidden_dim, device=device)
        agg_peak = torch.zeros(n_peaks, self.hidden_dim, device=device)

        # Edge 0: Spot-Spot
        if edge_index_ss is not None and edge_index_ss.size(1) > 0:
            src, dst = edge_index_ss[0], edge_index_ss[1]
            agg_spot += self._attn_aggregate(q_spot, k_spot, v_spot, src, dst, dist=dist_ss, num_nodes=n_spots)

        # Edge 1: Spot ↔ Gene
        if edge_index_sg is not None and edge_index_sg.size(1) > 0:
            src, dst = edge_index_sg[0], edge_index_sg[1]
            s2g, g2s = node_type[src] == 0, node_type[src] == 1
            if s2g.any():
                agg_gene += self._attn_aggregate(q_gene, k_spot, v_spot, src[s2g], dst[s2g] - n_spots, num_nodes=n_genes)
            if g2s.any():
                agg_spot += self._attn_aggregate(q_spot, k_gene, v_gene, src[g2s] - n_spots, dst[g2s], num_nodes=n_spots)

        # Edge 2: Spot ↔ Peak
        if edge_index_sp is not None and edge_index_sp.size(1) > 0:
            src, dst = edge_index_sp[0], edge_index_sp[1]
            s2p, p2s = node_type[src] == 0, node_type[src] == 2
            if s2p.any():
                agg_peak += self._attn_aggregate(q_peak, k_spot, v_spot, src[s2p], dst[s2p] - n_spots - n_genes, num_nodes=n_peaks)
            if p2s.any():
                agg_spot += self._attn_aggregate(q_spot, k_peak, v_peak, src[p2s] - n_spots - n_genes, dst[p2s], num_nodes=n_spots)

        # Edge 3: Gene ↔ Peak
        if edge_index_gp is not None and edge_index_gp.size(1) > 0:
            src, dst = edge_index_gp[0], edge_index_gp[1]
            g2p, p2g = node_type[src] == 1, node_type[src] == 2
            if g2p.any():
                agg_peak += self._attn_aggregate(q_peak, k_gene, v_gene, src[g2p] - n_spots, dst[g2p] - n_spots - n_genes, num_nodes=n_peaks)
            if p2g.any():
                agg_gene += self._attn_aggregate(q_gene, k_peak, v_peak, src[p2g] - n_spots - n_genes, dst[p2g] - n_spots, num_nodes=n_genes)

        h_new = h.clone()
        h_new[spot_mask] = self.norm_spot(h_spot + self.drop(self.update_spot(self.out_proj(agg_spot))))
        h_new[gene_mask] = self.norm_gene(h_gene + self.drop(self.update_gene(self.out_proj(agg_gene))))
        h_new[peak_mask] = self.norm_peak(h_peak + self.drop(self.update_peak(self.out_proj(agg_peak))))
        return h_new


class SupervisedEncoder(nn.Module):
    def __init__(self, n_spots, n_genes, n_peaks,
                 rna_dim, atac_dim, gene_in_dim, peak_in_dim,
                 hidden_dim=256, pos_dim=64, num_heads=4,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_spots, self.n_genes, self.n_peaks = n_spots, n_genes, n_peaks
        self.pos_enc = FourierPositionEncoding(coord_dim=2, out_dim=pos_dim)

        self.spot_rna_proj = nn.Sequential(
            nn.Linear(rna_dim, 128), nn.ReLU(), nn.Dropout(dropout))
        self.spot_atac_proj = nn.Sequential(
            nn.Linear(atac_dim, 128), nn.ReLU(), nn.Dropout(dropout))
        self.spot_fuse = nn.Linear(128 + 128 + pos_dim, hidden_dim)
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.peak_proj = nn.Sequential(
            nn.Linear(peak_in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        self.convs = nn.ModuleList([
            SpatialHeteroConv(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(n_layers)])

        self.rna_view_proj = nn.Linear(hidden_dim, 128)
        self.atac_view_proj = nn.Linear(hidden_dim, 128)

    def get_initial_embeddings(self, data, batch_size=4000):
        device = next(self.parameters()).device
        spot_rna, spot_atac = data['spot_rna'], data['spot_atac']
        gene_feat, peak_feat = data['gene_feat'], data['peak_feat']

        if device.type == 'cuda' and not spot_rna.is_cuda:
            spot_rna = spot_rna.to(device); spot_atac = spot_atac.to(device)
            gene_feat = gene_feat.to(device); peak_feat = peak_feat.to(device)
        coords = data['coords']
        if device.type == 'cuda' and not coords.is_cuda:
            coords = coords.to(device)

        h_pos = self.pos_enc(coords)

        n_spots = spot_rna.size(0)
        h_spot = torch.zeros(n_spots, self.hidden_dim, device=device)
        for s in range(0, n_spots, batch_size):
            e = min(s + batch_size, n_spots)
            hr = self.spot_rna_proj(spot_rna[s:e])
            ha = self.spot_atac_proj(spot_atac[s:e])
            h_spot[s:e] = self.spot_fuse(torch.cat([hr, ha, h_pos[s:e]], dim=-1))

        n_g = gene_feat.size(0); n_p = peak_feat.size(0)
        h_gene = torch.zeros(n_g, self.hidden_dim, device=device)
        h_peak = torch.zeros(n_p, self.hidden_dim, device=device)
        for s in range(0, max(n_g, n_p), batch_size):
            if s < n_g:
                e = min(s + batch_size, n_g)
                h_gene[s:e] = self.gene_proj(gene_feat[s:e])
            if s < n_p:
                e = min(s + batch_size, n_p)
                h_peak[s:e] = self.peak_proj(peak_feat[s:e])

        return torch.cat([h_spot, h_gene, h_peak], dim=0)

    def forward(self, h, edge_index_ss, edge_index_sg, edge_index_sp,
                edge_index_gp, dist_ss, node_type):
        for conv in self.convs:
            h = conv(h, edge_index_ss, edge_index_sg, edge_index_sp,
                     edge_index_gp, dist_ss, node_type,
                     self.n_spots, self.n_genes, self.n_peaks)
        return h

    def get_views(self, h, edge_index_sg, edge_index_sp, node_type):
        spot_mask = node_type == 0
        spot_h = h[spot_mask]
        ns = spot_h.size(0)

        rna_agg = torch.zeros_like(spot_h)
        if edge_index_sg is not None and edge_index_sg.size(1) > 0:
            src, dst = edge_index_sg[0], edge_index_sg[1]
            g2s = node_type[src] == 1
            if g2s.any():
                rna_agg.index_add_(0, dst[g2s], h[src[g2s]])
                deg = torch.zeros(ns, device=h.device)
                deg.index_add_(0, dst[g2s], torch.ones(g2s.sum(), device=h.device))
                valid = deg > 0
                rna_agg[valid] = rna_agg[valid] / deg[valid].unsqueeze(-1)
        rna_view = self.rna_view_proj(spot_h + rna_agg)

        atac_agg = torch.zeros_like(spot_h)
        if edge_index_sp is not None and edge_index_sp.size(1) > 0:
            src, dst = edge_index_sp[0], edge_index_sp[1]
            p2s = node_type[src] == 2
            if p2s.any():
                atac_agg.index_add_(0, dst[p2s], h[src[p2s]])
                deg = torch.zeros(ns, device=h.device)
                deg.index_add_(0, dst[p2s], torch.ones(p2s.sum(), device=h.device))
                valid = deg > 0
                atac_agg[valid] = atac_agg[valid] / deg[valid].unsqueeze(-1)
        atac_view = self.atac_view_proj(spot_h + atac_agg)
        return rna_view, atac_view


# ============================================================
# 2. Cluster Heads
# ============================================================

class ClusterHeads(nn.Module):
    """Per-time-point × per-modality classification heads."""

    def __init__(self, hidden_dim, n_rna_classes, n_atac_classes):
        super().__init__()
        self.rna_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n) if n > 0 else nn.Identity()
            for n in n_rna_classes])
        self.atac_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n) if n > 0 else nn.Identity()
            for n in n_atac_classes])

    def forward(self, spot_emb, time_labels):
        rna_logits, atac_logits = [], []
        for t in range(3):
            mask = time_labels == t
            rna_logits.append(
                self.rna_heads[t](spot_emb[mask]) if mask.any() and isinstance(self.rna_heads[t], nn.Linear) else None)
            atac_logits.append(
                self.atac_heads[t](spot_emb[mask]) if mask.any() and isinstance(self.atac_heads[t], nn.Linear) else None)
        return rna_logits, atac_logits


# ============================================================
# 3. Loss Functions (same as unsupervised + cluster losses)
# ============================================================

def recon_loss(h_spot, h_gene, h_peak, data, n_samples=30000):
    device = h_spot.device
    sg_loss = torch.tensor(0.0, device=device)
    sg_total = data['sg_gene_idx'].size(0)
    if sg_total > 0:
        n = min(n_samples, sg_total)
        idx = torch.randperm(sg_total, device=device)[:n]
        pred = (h_gene[data['sg_gene_idx'][idx]] * h_spot[data['sg_spot_idx'][idx]]).sum(dim=-1)
        sg_loss = F.mse_loss(pred, data['sg_true'][idx].to(device))

    sp_loss = torch.tensor(0.0, device=device)
    sp_total = data['sp_peak_idx'].size(0)
    if sp_total > 0:
        n = min(n_samples, sp_total)
        idx = torch.randperm(sp_total, device=device)[:n]
        pred = (h_peak[data['sp_peak_idx'][idx]] * h_spot[data['sp_spot_idx'][idx]]).sum(dim=-1)
        sp_loss = F.mse_loss(pred, data['sp_true'][idx].to(device))
    return sg_loss + sp_loss


def align_loss(rna_view, atac_view, temperature=0.07):
    rna = F.normalize(rna_view, dim=-1)
    atac = F.normalize(atac_view, dim=-1)
    sim = rna @ atac.T / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def spatial_loss(h_spot, edge_ss, dist_ss):
    if edge_ss is None or edge_ss.size(1) == 0:
        return torch.tensor(0.0, device=h_spot.device)
    diff = (h_spot[edge_ss[0]] - h_spot[edge_ss[1]]).pow(2).sum(dim=-1)
    return (torch.exp(-dist_ss) * diff).mean()


def gp_edge_loss(h_gene, h_peak, edge_gp, node_type, n_spots, n_genes, n_neg=5):
    if edge_gp is None or edge_gp.size(1) == 0:
        return torch.tensor(0.0, device=h_gene.device)
    device = h_gene.device
    src, dst = edge_gp[0], edge_gp[1]
    scores = []
    g2p, p2g = node_type[src] == 1, node_type[src] == 2
    if g2p.any():
        scores.append((h_gene[src[g2p] - n_spots] * h_peak[dst[g2p] - n_spots - n_genes]).sum(dim=-1))
    if p2g.any():
        scores.append((h_gene[dst[p2g] - n_spots] * h_peak[src[p2g] - n_spots - n_genes]).sum(dim=-1))
    if not scores:
        return torch.tensor(0.0, device=device)
    pos = torch.cat(scores)
    n_pos = pos.size(0)
    neg = (h_gene[torch.randint(0, h_gene.size(0), (n_pos * n_neg,), device=device)] *
           h_peak[torch.randint(0, h_peak.size(0), (n_pos * n_neg,), device=device)]).sum(dim=-1)
    all_s = torch.cat([pos, neg])
    all_t = torch.cat([torch.ones(n_pos, device=device), torch.zeros(n_pos * n_neg, device=device)])
    return F.binary_cross_entropy_with_logits(all_s, all_t)


# ============================================================
# 4. Trainer
# ============================================================

class SupervisedTrainer:
    def __init__(self, model, data, device='cuda',
                 lr=1e-3, wd=1e-4,
                 lambda_recon=1.0, lambda_align=0.3, lambda_spatial=0.2, lambda_gp=1.0,
                 lambda_rna=0.5, lambda_atac=0.5,
                 temperature=0.07, use_amp=False):
        self.model = model.to(device)
        self.data = data
        self.device = device

        self.lambda_recon = lambda_recon
        self.lambda_align = lambda_align
        self.lambda_spatial = lambda_spatial
        self.lambda_gp = lambda_gp
        self.lambda_rna = lambda_rna
        self.lambda_atac = lambda_atac
        self.temperature = temperature
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler() if self.use_amp else None

        self.cluster_heads = ClusterHeads(
            model.hidden_dim, data['n_rna_cls'], data['n_atac_cls']).to(device)

        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(self.cluster_heads.parameters()),
            lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=300)

        self.best_nmi = -1
        self.best_ari = -1

    def train_step(self, h0):
        model, data, device = self.model, self.data, self.device

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            h = model(h0, data['edge_index_ss'], data['edge_index_sg'],
                      data['edge_index_sp'], data['edge_index_gp'],
                      data['dist_ss'], data['node_type'])
            h_spot = h[:model.n_spots]
            h_gene = h[model.n_spots:model.n_spots + model.n_genes]
            h_peak = h[model.n_spots + model.n_genes:]

            loss_recon = recon_loss(h_spot, h_gene, h_peak, data)
            rna_view, atac_view = model.get_views(
                h, data['edge_index_sg'], data['edge_index_sp'], data['node_type'])
            loss_align = align_loss(rna_view, atac_view, self.temperature)
            loss_spatial = spatial_loss(h_spot, data['edge_index_ss'], data['dist_ss'])
            loss_gp = gp_edge_loss(h_gene, h_peak, data['edge_index_gp'],
                                   data['node_type'], model.n_spots, model.n_genes)

            # Cluster supervision
            rna_logits, atac_logits = self.cluster_heads(h_spot, data['time_labels'])
            loss_rna = torch.tensor(0.0, device=device)
            loss_atac = torch.tensor(0.0, device=device)
            tl = data['time_labels']
            for t in range(3):
                mask = tl == t
                if mask.any() and rna_logits[t] is not None:
                    valid = data['rna_labels'][mask] >= 0
                    if valid.any():
                        loss_rna += F.cross_entropy(rna_logits[t][valid], data['rna_labels'][mask][valid])
                if mask.any() and atac_logits[t] is not None:
                    valid = data['atac_labels'][mask] >= 0
                    if valid.any():
                        loss_atac += F.cross_entropy(atac_logits[t][valid], data['atac_labels'][mask][valid])

            total = (self.lambda_recon * loss_recon +
                     self.lambda_align * loss_align +
                     self.lambda_spatial * loss_spatial +
                     self.lambda_gp * loss_gp +
                     self.lambda_rna * loss_rna +
                     self.lambda_atac * loss_atac)

        return total, {
            'recon': loss_recon.item(), 'align': loss_align.item(),
            'spatial': loss_spatial.item(), 'gp': loss_gp.item(),
            'rna': loss_rna.item(), 'atac': loss_atac.item(),
            'total': total.item(),
        }

    def evaluate(self, h_spot):
        mask = self.data['eval_mask']
        if mask.sum() < 2:
            return {'nmi': 0.0, 'ari': 0.0}
        emb = h_spot[mask].detach().cpu().numpy()
        labels = self.data['joint_labels'][mask].cpu().numpy()
        n_clusters = len(np.unique(labels))
        if n_clusters >= 2:
            pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(emb)
            return {'nmi': normalized_mutual_info_score(labels, pred),
                    'ari': adjusted_rand_score(labels, pred)}
        return {'nmi': 0.0, 'ari': 0.0}

    def train(self, epochs=200, eval_every=5):
        data = self.data
        print("Computing initial node embeddings...")
        self.model.eval()
        with torch.no_grad():
            h0 = self.model.get_initial_embeddings(data)
        print(f"  h0: {h0.shape}, device: {h0.device}")

        history = []
        for epoch in range(epochs):
            self.model.train()
            self.cluster_heads.train()
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    total_loss, losses = self.train_step(h0)
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(self.cluster_heads.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss, losses = self.train_step(h0)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(self.cluster_heads.parameters(), 5.0)
                self.optimizer.step()

            self.scheduler.step()
            metrics = {'epoch': epoch + 1, **losses}

            if (epoch + 1) % eval_every == 0 or epoch == 0 or epoch == epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    h = self.model(h0, data['edge_index_ss'], data['edge_index_sg'],
                                   data['edge_index_sp'], data['edge_index_gp'],
                                   data['dist_ss'], data['node_type'])
                    eval_m = self.evaluate(h[:self.model.n_spots])
                    metrics.update(eval_m)
                    if eval_m['nmi'] > self.best_nmi: self.best_nmi = eval_m['nmi']
                    if eval_m['ari'] > self.best_ari: self.best_ari = eval_m['ari']

            history.append(metrics)
            if (epoch + 1) % 10 == 0 or epoch < 10:
                print(f"Epoch {epoch+1:4d} | Loss {losses['total']:.4f} | "
                      f"R={losses['recon']:.3f} A={losses['align']:.3f} "
                      f"S={losses['spatial']:.3f} GP={losses['gp']:.3f} "
                      f"r={losses['rna']:.3f} a={losses['atac']:.3f} | "
                      f"NMI={metrics.get('nmi', 0):.4f} ARI={metrics.get('ari', 0):.4f}")

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
        return {'spot': h[:ns].detach().cpu().numpy(),
                'gene': h[ns:ns + ng].detach().cpu().numpy(),
                'peak': h[ns + ng:].detach().cpu().numpy()}

    def compute_gp_scores(self):
        self.model.eval()
        with torch.no_grad():
            h0 = self.model.get_initial_embeddings(self.data)
            h = self.model(h0, self.data['edge_index_ss'], self.data['edge_index_sg'],
                           self.data['edge_index_sp'], self.data['edge_index_gp'],
                           self.data['dist_ss'], self.data['node_type'])
        ns, ng = self.model.n_spots, self.model.n_genes
        h_gene = h[ns:ns + ng]
        h_peak = h[ns + ng:]
        return torch.sigmoid(h_gene @ h_peak.T).detach().cpu().numpy()

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'cluster_heads': self.cluster_heads.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, path)
        print(f"Saved to {path}")
