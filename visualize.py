"""Visualization for trained spot embeddings (works for sup or unsup).

Inputs:
  --emb_path   spot embedding .npy, shape (n_spots, hidden)
  --out_dir    where to save figures
  --label      short string used in plot titles (e.g. "Supervised", "Unsupervised")
  --resolution Leiden resolution (default 1.0)

Outputs (under out_dir):
  umap_by_time.png      UMAP-2D coloured by time point
  umap_by_cluster.png   UMAP-2D coloured by Leiden cluster
  spatial_clusters.png  3 panels (E13/P21/P22), spots at (x,y) coloured by cluster
  leiden_clusters.npy   cluster assignments
  umap_2d.npy           UMAP 2D coordinates

Run with HGNA env:
  /home/altedfish/anaconda3/envs/HGNA/bin/python visualize_sup.py \
      --emb_path output_sup_joint/spot_emb.npy --out_dir output_sup_joint/figs \
      --label Supervised
"""
import os, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import scanpy as sc
import anndata as ad
import umap
import matplotlib.pyplot as plt

sc.settings.verbosity = 0

BASE = os.path.dirname(os.path.abspath(__file__))
COORDS_PATH = os.path.join(BASE, 'graph_combined', 'coords.npy')
TIME_PATH = os.path.join(BASE, 'graph_combined', 'spot_time_labels.npy')

TIME_NAMES = ['E13', 'P21', 'P22']
TIME_COLORS = {'E13': '#1f77b4', 'P21': '#2ca02c', 'P22': '#d62728'}


def leiden_cluster(emb, resolution=1.0, n_neighbors=15, seed=42):
    a = ad.AnnData(emb.astype(np.float32))
    sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.leiden(a, resolution=resolution, random_state=seed,
                 flavor='igraph', directed=False, n_iterations=2)
    return a.obs['leiden'].astype(int).values


def umap_2d(emb, n_neighbors=15, min_dist=0.3, seed=42):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=seed, metric='euclidean')
    return reducer.fit_transform(emb)


def cluster_palette(n_clusters, seed=0):
    if n_clusters <= 20:
        cmap = plt.get_cmap('tab20', n_clusters)
        return [cmap(i) for i in range(n_clusters)]
    # fallback: HSV evenly
    return [plt.get_cmap('hsv')(i / n_clusters) for i in range(n_clusters)]


def plot_umap_by_time(coords2d, time_labels, out_path, label):
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=150)
    for t, tn in enumerate(TIME_NAMES):
        m = time_labels == t
        ax.scatter(coords2d[m, 0], coords2d[m, 1], s=2, alpha=0.6,
                   c=TIME_COLORS[tn], label=f'{tn} (n={m.sum()})', linewidths=0)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_title(f'{label} spot embedding (UMAP) — coloured by time point')
    ax.legend(loc='best', frameon=True, markerscale=4)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f'  wrote {out_path}')


def plot_umap_by_cluster(coords2d, cluster_labels, out_path, label):
    n_c = int(cluster_labels.max()) + 1
    palette = cluster_palette(n_c)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    for k in range(n_c):
        m = cluster_labels == k
        ax.scatter(coords2d[m, 0], coords2d[m, 1], s=2, alpha=0.7,
                   c=[palette[k]], label=f'{k}', linewidths=0)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_title(f'{label} spot embedding (UMAP) — coloured by Leiden cluster (k={n_c})')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=8, markerscale=3, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


def plot_spatial_clusters(coords, time_labels, cluster_labels, out_path, label):
    n_c = int(cluster_labels.max()) + 1
    palette = cluster_palette(n_c)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=150)
    for ax, t, tn in zip(axes, range(3), TIME_NAMES):
        m = time_labels == t
        ax.scatter(coords[m, 0], coords[m, 1], s=4,
                   c=[palette[k] for k in cluster_labels[m]],
                   linewidths=0)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'{tn}  (n={m.sum()})')
        ax.set_aspect('equal', adjustable='datalim')
        ax.invert_yaxis()
    handles = [plt.Line2D([0], [0], marker='o', linestyle='',
                          markerfacecolor=palette[k], markeredgecolor='none',
                          markersize=6, label=f'{k}') for k in range(n_c)]
    fig.legend(handles=handles, loc='center right', frameon=False, fontsize=8,
               bbox_to_anchor=(1.0, 0.5), ncol=1, title='Leiden cluster')
    fig.suptitle(f'{label} model — spatial Leiden clusters per time point',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.93, 0.96])
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb_path', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--label', default='Model')
    ap.add_argument('--resolution', type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading...')
    emb = np.load(args.emb_path)
    coords = np.load(COORDS_PATH)
    time_labels = np.load(TIME_PATH)
    print(f'  emb {emb.shape}  coords {coords.shape}  time {time_labels.shape}')

    print(f'Running Leiden (joint over all spots, resolution={args.resolution})...')
    cluster_labels = leiden_cluster(emb, resolution=args.resolution)
    n_c = int(cluster_labels.max()) + 1
    print(f'  → {n_c} clusters; sizes: '
          + ', '.join(f'{k}:{(cluster_labels==k).sum()}' for k in range(n_c)))

    print('Running UMAP on embedding (this can take a minute)...')
    coords2d = umap_2d(emb)
    print(f'  → UMAP shape {coords2d.shape}')

    print('Plotting...')
    plot_umap_by_time(coords2d, time_labels,
                      os.path.join(args.out_dir, 'umap_by_time.png'), args.label)
    plot_umap_by_cluster(coords2d, cluster_labels,
                         os.path.join(args.out_dir, 'umap_by_cluster.png'), args.label)
    plot_spatial_clusters(coords, time_labels, cluster_labels,
                          os.path.join(args.out_dir, 'spatial_clusters.png'), args.label)

    np.save(os.path.join(args.out_dir, 'leiden_clusters.npy'), cluster_labels)
    np.save(os.path.join(args.out_dir, 'umap_2d.npy'), coords2d)
    print('\nAll figures saved under', args.out_dir)


if __name__ == '__main__':
    main()
