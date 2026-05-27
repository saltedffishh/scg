"""Re-evaluate trained spot embeddings with Leiden clustering.

Replaces the KMeans step in evaluate() with scanpy's Leiden (KNN graph + modularity).
Reports per-time-point metrics under three clustering pools (mirroring training-time
cluster_mode): per_time, joint_all, joint_e13p21.

Run with HGNA env (has scanpy + leidenalg):
  /home/altedfish/anaconda3/envs/HGNA/bin/python eval_leiden.py
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.metrics import (normalized_mutual_info_score, adjusted_mutual_info_score,
    adjusted_rand_score, fowlkes_mallows_score, mutual_info_score)
from scipy.optimize import linear_sum_assignment

sc.settings.verbosity = 0

BASE = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(BASE, 'graph_combined')
TIME_NAMES = ['E13', 'P21', 'P22']
METRIC_KEYS = ['nmi', 'ami', 'ari', 'fmi', 'mi', 'acc']


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64)
    D = int(max(y_pred.max(), y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return w[row, col].sum() / y_pred.size


def all_metrics(labels, pred):
    return {
        'nmi': normalized_mutual_info_score(labels, pred),
        'ami': adjusted_mutual_info_score(labels, pred),
        'ari': adjusted_rand_score(labels, pred),
        'fmi': fowlkes_mallows_score(labels, pred),
        'mi':  mutual_info_score(labels, pred),
        'acc': cluster_acc(labels, pred),
    }


def load_labels():
    meta = pd.read_csv(os.path.join(GRAPH_DIR, 'meta.csv'))
    spot_time_labels = np.load(os.path.join(GRAPH_DIR, 'spot_time_labels.npy'))
    n_spots = len(spot_time_labels)
    joint_labels = np.full(n_spots, -1, dtype=np.int64)
    for t, tn in enumerate(TIME_NAMES):
        meta_mask = meta['time_label'].values == tn
        spot_indices = np.where(spot_time_labels == t)[0]
        if 'Joint_clusters' not in meta.columns:
            continue
        col = meta.loc[meta_mask, 'Joint_clusters']
        nonnull = col.dropna()
        if nonnull.empty:
            continue
        unique_joint = sorted(set(nonnull.values))
        jmap = {v: i for i, v in enumerate(unique_joint)}
        for si, val in zip(spot_indices, col.values):
            if pd.notna(val):
                joint_labels[si] = jmap[val]
    return spot_time_labels, joint_labels


def leiden_on(emb, resolution=1.0, n_neighbors=15, seed=42):
    a = ad.AnnData(emb.astype(np.float32))
    sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.leiden(a, resolution=resolution, random_state=seed,
                 flavor='igraph', directed=False, n_iterations=2)
    return a.obs['leiden'].astype(int).values


def eval_one(emb, time_labels, joint_labels, mode, resolution=1.0, n_neighbors=15):
    out = {}
    agg = {k: [] for k in METRIC_KEYS}
    n_clusters_info = {}

    if mode == 'per_time':
        for t, tn in enumerate(TIME_NAMES):
            mask = (time_labels == t) & (joint_labels >= 0)
            if mask.sum() < 2:
                continue
            sub_emb = emb[mask]
            pred = leiden_on(sub_emb, resolution=resolution, n_neighbors=n_neighbors)
            n_clusters_info[tn] = len(np.unique(pred))
            m = all_metrics(joint_labels[mask], pred)
            for k, v in m.items():
                out[f'{k}_{tn}'] = v; agg[k].append(v)
    else:
        if mode == 'joint_all':
            pool_mask = np.ones_like(time_labels, dtype=bool)
        else:  # joint_e13p21
            pool_mask = (time_labels == 0) | (time_labels == 1)
        pool_idx = np.where(pool_mask)[0]
        pred_pool = leiden_on(emb[pool_idx], resolution=resolution, n_neighbors=n_neighbors)
        n_clusters_info['pool'] = len(np.unique(pred_pool))
        pred_all = np.full(emb.shape[0], -1, dtype=np.int64)
        pred_all[pool_idx] = pred_pool
        for t, tn in enumerate(TIME_NAMES):
            mask = (time_labels == t) & (joint_labels >= 0) & (pred_all >= 0)
            if mask.sum() < 2:
                continue
            m = all_metrics(joint_labels[mask], pred_all[mask])
            for k, v in m.items():
                out[f'{k}_{tn}'] = v; agg[k].append(v)

    for k in METRIC_KEYS:
        out[k] = float(np.mean(agg[k])) if agg[k] else 0.0
    out['_n_clusters'] = n_clusters_info
    return out


def fmt_row(name, m):
    keys = ['nmi', 'ami', 'ari', 'fmi', 'mi', 'acc']
    vals = '  '.join(f'{k.upper()}={m.get(k,0):.4f}' for k in keys)
    return f'  {name:5s} | {vals}'


def report(emb_path, label, time_labels, joint_labels, resolution=1.0, n_neighbors=15):
    print(f'\n{"="*78}')
    print(f' {label}')
    print(f'   emb={emb_path}  resolution={resolution}  n_neighbors={n_neighbors}')
    print(f'{"="*78}')
    emb = np.load(emb_path)
    for mode in ['per_time', 'joint_e13p21', 'joint_all']:
        m = eval_one(emb, time_labels, joint_labels, mode,
                     resolution=resolution, n_neighbors=n_neighbors)
        print(f'\n[mode={mode}]  n_clusters={m["_n_clusters"]}')
        for tn in TIME_NAMES:
            if f'nmi_{tn}' in m:
                row = {k: m[f'{k}_{tn}'] for k in METRIC_KEYS}
                print(fmt_row(tn, row))
        avg = {k: m[k] for k in METRIC_KEYS}
        print(fmt_row('avg', avg))


def main():
    time_labels, joint_labels = load_labels()
    print(f'spots: {len(time_labels)}, joint-labelled: {(joint_labels>=0).sum()}')

    runs = [
        (os.path.join(BASE, 'output_sup_joint', 'spot_emb.npy'),
         'supervised (joint_all 250ep, hidden=128) [embedding only — eval here]'),
        (os.path.join(BASE, 'output_unsup_joint_e13p21', 'spot_emb.npy'),
         'unsupervised (joint_e13p21 250ep, hidden=128) [embedding only — eval here]'),
    ]
    for resolution in [0.5, 1.0, 1.5]:
        print(f'\n\n##############  RESOLUTION = {resolution}  ##############')
        for emb_path, label in runs:
            if not os.path.exists(emb_path):
                print(f'skip (missing): {emb_path}')
                continue
            report(emb_path, label, time_labels, joint_labels,
                   resolution=resolution, n_neighbors=15)


if __name__ == '__main__':
    main()
