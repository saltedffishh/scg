"""
Shared data loader for both supervised and unsupervised training.
Pre-computes edge expression values for fast GPU training.

Node ID layout:
  Spot: [0, n_spots)
  Gene: [n_spots, n_spots+n_genes)
  Peak: [n_spots+n_genes, n_spots+n_genes+n_peaks)

Edge types:
  0: Spot-Spot (KNN)
  1: Spot-Gene (bidirectional)
  2: Spot-Peak (bidirectional)
  3: Gene-Peak (bidirectional)
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
import pandas as pd


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_combined_data(data_dir, device=None, use_svd=True, svd_dim=100):
    """
    Load graph_combined/ data with pre-computed edge values for GPU training.

    Args:
        data_dir: path to graph_combined/
        device: torch device (auto-detect if None)
        use_svd: use TruncatedSVD feature reduction (recommended)
        svd_dim: number of SVD components per modality

    Returns:
        dict with all tensors needed for training.
    """
    if device is None:
        device = get_device()

    print(f"Loading graph data... (device: {device})")

    # ---- Load graph structure ----
    g = np.load(os.path.join(data_dir, 'graph.npz'), allow_pickle=True)
    n_spots = int(g['n_spots'])
    n_genes = int(g['n_genes'])
    n_peaks = int(g['n_peaks'])
    spot_time_labels = np.load(os.path.join(data_dir, 'spot_time_labels.npy'))
    coords = np.load(os.path.join(data_dir, 'coords.npy'))

    # ---- Node types ----
    node_type = np.zeros(n_spots + n_genes + n_peaks, dtype=np.int64)
    node_type[n_spots:n_spots + n_genes] = 1
    node_type[n_spots + n_genes:] = 2
    node_type_t = torch.tensor(node_type, dtype=torch.long, device=device)

    # ---- Edges (split by type, keep on device) ----
    edge_index_ss = torch.tensor(g['spot_spot_edges'], dtype=torch.long, device=device)
    edge_index_sg = torch.tensor(g['spot_gene_edges'], dtype=torch.long, device=device)
    edge_index_sp = torch.tensor(g['spot_peak_edges'], dtype=torch.long, device=device)
    edge_index_gp = torch.tensor(g['gene_peak_edges'], dtype=torch.long, device=device)

    # ---- Spot-Spot distances ----
    ss_src = g['spot_spot_edges'][0]
    ss_dst = g['spot_spot_edges'][1]
    ss_dist = np.sqrt(((coords[ss_src] - coords[ss_dst]) ** 2).sum(axis=1))
    dist_ss = torch.tensor(ss_dist, dtype=torch.float32, device=device)

    # ---- Time labels ----
    time_labels_t = torch.tensor(spot_time_labels, dtype=torch.long, device=device)

    # ---- Coordinates ----
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    # ---- Expression matrices (sparse CSR, kept on CPU for index ops) ----
    rna_matrix = sp.load_npz(os.path.join(data_dir, 'rna_matrix.npz'))
    atac_matrix = sp.load_npz(os.path.join(data_dir, 'atac_matrix.npz'))

    # ---- SVD feature reduction ----
    if use_svd:
        print(f"  Running TruncatedSVD (dim={svd_dim})...")
        print(f"    Spot RNA (13775 x 10192)...", end=" ", flush=True)
        svd_rna_spot = TruncatedSVD(n_components=svd_dim, n_iter=5, random_state=42)
        spot_rna_feat = svd_rna_spot.fit_transform(rna_matrix.T).astype(np.float32)
        print(f"explained={svd_rna_spot.explained_variance_ratio_.sum():.3f}")

        print(f"    Spot ATAC (13775 x 16864)...", end=" ", flush=True)
        svd_atac_spot = TruncatedSVD(n_components=svd_dim, n_iter=5, random_state=42)
        spot_atac_feat = svd_atac_spot.fit_transform(atac_matrix.T).astype(np.float32)
        print(f"explained={svd_atac_spot.explained_variance_ratio_.sum():.3f}")

        print(f"    Gene (10192 x 13775)...", end=" ", flush=True)
        svd_gene = TruncatedSVD(n_components=svd_dim, n_iter=5, random_state=42)
        gene_feat = svd_gene.fit_transform(rna_matrix).astype(np.float32)
        print(f"explained={svd_gene.explained_variance_ratio_.sum():.3f}")

        print(f"    Peak (16864 x 13775)...", end=" ", flush=True)
        svd_peak = TruncatedSVD(n_components=svd_dim, n_iter=5, random_state=42)
        peak_feat = svd_peak.fit_transform(atac_matrix).astype(np.float32)
        print(f"explained={svd_peak.explained_variance_ratio_.sum():.3f}")

        rna_in_dim = svd_dim
        atac_in_dim = svd_dim
        gene_in_dim = svd_dim
        peak_in_dim = svd_dim

        # Convert to tensors (can fit in GPU memory easily)
        spot_rna_t = torch.tensor(spot_rna_feat, dtype=torch.float32)  # (13775, D)
        spot_atac_t = torch.tensor(spot_atac_feat, dtype=torch.float32)
        gene_feat_t = torch.tensor(gene_feat, dtype=torch.float32)     # (10192, D)
        peak_feat_t = torch.tensor(peak_feat, dtype=torch.float32)     # (16864, D)
        use_sparse_features = False

    else:
        # Raw sparse features (not recommended for CPU)
        spot_rna_t = rna_matrix.T.tocsr()
        spot_atac_t = atac_matrix.T.tocsr()
        gene_feat_t = rna_matrix
        peak_feat_t = atac_matrix
        rna_in_dim = rna_matrix.shape[0]
        atac_in_dim = atac_matrix.shape[0]
        gene_in_dim = rna_matrix.shape[1]
        peak_in_dim = atac_matrix.shape[1]
        use_sparse_features = True

    # ---- Pre-compute edge expression values for reconstruction loss ----
    # This avoids expensive sparse lookups during training.
    print("  Pre-computing edge expression values...")

    # Spot-Gene edges (gene→spot direction: gene=source, spot=target)
    sg_src = g['spot_gene_edges'][0]
    sg_dst = g['spot_gene_edges'][1]
    sg_g2s = node_type[sg_src] == 1  # gene→spot mask
    sg_g2s_idx = np.where(sg_g2s)[0]
    sg_gene_local = sg_src[sg_g2s_idx] - n_spots  # local gene index
    sg_spot_local = sg_dst[sg_g2s_idx]            # local spot index
    print(f"    Spot-Gene (gene→spot): {len(sg_g2s_idx):,} edges")

    # Collect true expression values
    sg_true_vals = np.array(
        [rna_matrix[g, s] for g, s in zip(sg_gene_local, sg_spot_local)],
        dtype=np.float32)
    sg_gene_idx_t = torch.tensor(sg_gene_local, dtype=torch.long, device=device)
    sg_spot_idx_t = torch.tensor(sg_spot_local, dtype=torch.long, device=device)
    sg_true_t = torch.tensor(sg_true_vals, dtype=torch.float32, device=device)

    # Spot-Peak edges (peak→spot direction)
    sp_src = g['spot_peak_edges'][0]
    sp_dst = g['spot_peak_edges'][1]
    sp_p2s = node_type[sp_src] == 2
    sp_p2s_idx = np.where(sp_p2s)[0]
    sp_peak_local = sp_src[sp_p2s_idx] - n_spots - n_genes
    sp_spot_local = sp_dst[sp_p2s_idx]
    print(f"    Spot-Peak (peak→spot): {len(sp_p2s_idx):,} edges")

    sp_true_vals = np.array(
        [atac_matrix[p, s] for p, s in zip(sp_peak_local, sp_spot_local)],
        dtype=np.float32)
    sp_peak_idx_t = torch.tensor(sp_peak_local, dtype=torch.long, device=device)
    sp_spot_idx_t = torch.tensor(sp_spot_local, dtype=torch.long, device=device)
    sp_true_t = torch.tensor(sp_true_vals, dtype=torch.float32, device=device)

    # ---- Metadata with cluster labels ----
    meta = pd.read_csv(os.path.join(data_dir, 'meta.csv'))
    time_names_list = ['E13', 'P21', 'P22']

    # Encode cluster labels per time point
    rna_labels_all = np.full(n_spots, -1, dtype=np.int64)
    atac_labels_all = np.full(n_spots, -1, dtype=np.int64)
    joint_labels_all = np.full(n_spots, -1, dtype=np.int64)
    n_rna_cls = [0, 0, 0]
    n_atac_cls = [0, 0, 0]

    for t, tn in enumerate(time_names_list):
        mask = meta['time_label'].values == tn
        spot_indices = np.where(spot_time_labels == t)[0]

        # RNA clusters
        rna_col = meta.loc[mask, 'RNA_clusters'].values
        unique_rna = sorted(set(rna_col))
        n_rna_cls[t] = len(unique_rna)
        rna_map = {v: i for i, v in enumerate(unique_rna)}
        for si, val in zip(spot_indices, rna_col):
            rna_labels_all[si] = rna_map[val]

        # ATAC clusters
        atac_col = meta.loc[mask, 'ATAC_clusters'].values
        unique_atac = sorted(set(atac_col))
        n_atac_cls[t] = len(unique_atac)
        atac_map = {v: i for i, v in enumerate(unique_atac)}
        for si, val in zip(spot_indices, atac_col):
            atac_labels_all[si] = atac_map[val]

        # Joint clusters (E13+P21 only)
        if 'Joint_clusters' in meta.columns:
            joint_col = meta.loc[mask, 'Joint_clusters'].dropna()
            if len(joint_col) > 0:
                unique_joint = sorted(set(joint_col.values))
                joint_map = {v: i for i, v in enumerate(unique_joint)}
                for si, val in zip(spot_indices, joint_col.values):
                    joint_labels_all[si] = joint_map[val]

    rna_labels_t = torch.tensor(rna_labels_all, dtype=torch.long, device=device)
    atac_labels_t = torch.tensor(atac_labels_all, dtype=torch.long, device=device)
    joint_labels_t = torch.tensor(joint_labels_all, dtype=torch.long, device=device)

    # Evaluation mask: spots with valid Joint_clusters (E13+P21)
    eval_mask = joint_labels_all >= 0

    # ---- Summary ----
    print(f"\n  Nodes: {n_spots} spots + {n_genes} genes + {n_peaks} peaks = {n_spots+n_genes+n_peaks}")
    print(f"  Edges: SS={edge_index_ss.size(1):,}  SG={edge_index_sg.size(1):,}  "
          f"SP={edge_index_sp.size(1):,}  GP={edge_index_gp.size(1):,}")
    print(f"  RNA cls/TP: {n_rna_cls}  ATAC cls/TP: {n_atac_cls}")
    print(f"  Joint eval spots: {eval_mask.sum()} / {n_spots}")
    print(f"  Feature dims: rna={rna_in_dim} atac={atac_in_dim} gene={gene_in_dim} peak={peak_in_dim}")
    if device.type == 'cuda':
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    return {
        'n_spots': n_spots, 'n_genes': n_genes, 'n_peaks': n_peaks,
        'rna_in_dim': rna_in_dim, 'atac_in_dim': atac_in_dim,
        'gene_in_dim': gene_in_dim, 'peak_in_dim': peak_in_dim,
        'n_rna_cls': n_rna_cls, 'n_atac_cls': n_atac_cls,

        # Graph structure (on device)
        'edge_index_ss': edge_index_ss, 'edge_index_sg': edge_index_sg,
        'edge_index_sp': edge_index_sp, 'edge_index_gp': edge_index_gp,
        'dist_ss': dist_ss, 'node_type': node_type_t,
        'time_labels': time_labels_t, 'coords': coords_t,

        # Node features
        'spot_rna': spot_rna_t, 'spot_atac': spot_atac_t,
        'gene_feat': gene_feat_t, 'peak_feat': peak_feat_t,
        'use_sparse_features': use_sparse_features,

        # Pre-computed edge values for recon loss
        'sg_gene_idx': sg_gene_idx_t, 'sg_spot_idx': sg_spot_idx_t,
        'sg_true': sg_true_t,
        'sp_peak_idx': sp_peak_idx_t, 'sp_spot_idx': sp_spot_idx_t,
        'sp_true': sp_true_t,

        # Labels (for supervised version) and eval
        'rna_labels': rna_labels_t, 'atac_labels': atac_labels_t,
        'joint_labels': joint_labels_t, 'eval_mask': eval_mask,

        # Raw matrices (on CPU, for recon loss access)
        'rna_matrix': rna_matrix, 'atac_matrix': atac_matrix,
        'spot_time_labels': spot_time_labels,
    }


def to_gpu(data, device=None):
    """Move float/int tensors in data dict to GPU. Called before training."""
    if device is None:
        device = get_device()
    if device.type != 'cuda':
        return data  # nothing to do

    for key in data:
        if isinstance(data[key], torch.Tensor):
            if data[key].dtype in (torch.float32, torch.float16, torch.int64, torch.int32, torch.long, torch.bool):
                data[key] = data[key].to(device, non_blocking=True)
    return data
