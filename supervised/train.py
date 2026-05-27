"""
Supervised Training Entry Point
================================
Uses RNA_clusters + ATAC_clusters as training signals.
  L = L_recon + λ1·L_align + λ2·L_spatial + λ3·L_gp
      + λ4·L_rna_cluster + λ5·L_atac_cluster

Joint_clusters are HELD OUT for evaluation only.

Supports GPU (CUDA/MPS) with mixed precision.

Usage:
  python supervised/train.py                           # default params
  python supervised/train.py --epochs 300 --svd 200    # better quality
  python supervised/train.py --device cuda --amp        # GPU + mixed precision
"""

import argparse, os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.loader import load_combined_data, get_device
from supervised.model import SupervisedEncoder, SupervisedTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Supervised Spatial Hetero Graph Encoder')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'graph_combined'))
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--pos_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_align', type=float, default=0.3)
    parser.add_argument('--lambda_spatial', type=float, default=0.2)
    parser.add_argument('--lambda_gp', type=float, default=1.0)
    parser.add_argument('--lambda_rna', type=float, default=0.5)
    parser.add_argument('--lambda_atac', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--no_svd', action='store_true')
    parser.add_argument('--svd_dim', type=int, default=100)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision (GPU only)')
    parser.add_argument('--joint_cluster', action='store_true',
                        help='Cluster all spots (E13+P21+P22) jointly with one KMeans, '
                             'then evaluate per-time metrics on E13/P21 only.')
    parser.add_argument('--joint_cluster_e13p21', action='store_true',
                        help='Cluster only E13+P21 spots jointly with one KMeans, '
                             'then evaluate per-time metrics on E13/P21.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.joint_cluster and args.joint_cluster_e13p21:
        parser.error('--joint_cluster and --joint_cluster_e13p21 are mutually exclusive')
    cluster_mode = ('joint_e13p21' if args.joint_cluster_e13p21
                    else 'joint_all' if args.joint_cluster
                    else 'per_time')

    # Output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output_sup')
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # Config summary
    print("=" * 60)
    print("Supervised Spatial Heterogeneous Graph Encoder")
    print("=" * 60)
    print(f"  hidden={args.hidden}, layers={args.n_layers}, heads={args.num_heads}")
    print(f"  epochs={args.epochs}, lr={args.lr}, amp={args.amp}")
    print(f"  λ: recon={args.lambda_recon} align={args.lambda_align} "
          f"spatial={args.lambda_spatial} gp={args.lambda_gp} "
          f"rna={args.lambda_rna} atac={args.lambda_atac}")
    print(f"  svd={not args.no_svd} dim={args.svd_dim}")
    print(f"  cluster_mode={cluster_mode}")
    print(f"  [Joint_clusters HELD OUT for evaluation]")

    # Load data
    print("\n" + "-" * 40)
    data = load_combined_data(
        args.data_dir, device=device,
        use_svd=not args.no_svd, svd_dim=args.svd_dim)

    # Create model
    print("\n" + "-" * 40)
    print("Creating model (with cluster heads)...")
    model = SupervisedEncoder(
        n_spots=data['n_spots'], n_genes=data['n_genes'], n_peaks=data['n_peaks'],
        rna_dim=data['rna_in_dim'], atac_dim=data['atac_in_dim'],
        gene_in_dim=data['gene_in_dim'], peak_in_dim=data['peak_in_dim'],
        hidden_dim=args.hidden, pos_dim=args.pos_dim,
        num_heads=args.num_heads, n_layers=args.n_layers, dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Encoder params: {n_params:,}")

    # Trainer
    trainer = SupervisedTrainer(
        model=model, data=data, device=device,
        lr=args.lr, wd=args.wd,
        lambda_recon=args.lambda_recon, lambda_align=args.lambda_align,
        lambda_spatial=args.lambda_spatial, lambda_gp=args.lambda_gp,
        lambda_rna=args.lambda_rna, lambda_atac=args.lambda_atac,
        temperature=args.temperature, use_amp=args.amp,
        cluster_mode=cluster_mode,
    )

    # Train
    print("\n" + "=" * 60)
    history = trainer.train(epochs=args.epochs, eval_every=5)

    # Save model
    trainer.save(os.path.join(args.output_dir, 'model.pt'))

    # Save embeddings
    print("\nExtracting embeddings...")
    emb = trainer.get_embeddings()
    np.save(os.path.join(args.output_dir, 'spot_emb.npy'), emb['spot'])
    np.save(os.path.join(args.output_dir, 'gene_emb.npy'), emb['gene'])
    np.save(os.path.join(args.output_dir, 'peak_emb.npy'), emb['peak'])
    print(f"  spot  {emb['spot'].shape}")
    print(f"  gene  {emb['gene'].shape}")
    print(f"  peak  {emb['peak'].shape}")

    # Gene-Peak scores
    print("\nComputing Gene-Peak scores...")
    gp_scores = trainer.compute_gp_scores()
    np.save(os.path.join(args.output_dir, 'gp_scores.npy'), gp_scores)
    print(f"  gp_scores {gp_scores.shape}")

    # Save history and config
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        config_out = {k: str(v) if isinstance(v, torch.device) else v
                      for k, v in vars(args).items()}
        config_out['encoder_params'] = n_params
        json.dump(config_out, f, indent=2)

    print(f"\nDone! → {args.output_dir}")


if __name__ == '__main__':
    main()
