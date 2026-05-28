"""
Semi-Supervised Training Entry Point
====================================
Identical to supervised/train.py except a random fraction `--label_ratio`
of spots is used for L_rna / L_atac (other spots' labels masked to -1).

Usage:
  python semi_supervised/train.py --label_ratio 0.3 --epochs 250 --hidden 128 \
      --joint_cluster_e13p21 --device cuda --amp \
      --output_dir output_semi_r0.3
"""

import argparse, os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.loader import load_combined_data, get_device
from semi_supervised.model import SemiSupervisedTrainer, SupervisedEncoder


def main():
    parser = argparse.ArgumentParser(
        description='Semi-Supervised Spatial Hetero Graph Encoder')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'graph_combined'))
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--hidden', type=int, default=128)
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
    parser.add_argument('--joint_cluster', action='store_true')
    parser.add_argument('--joint_cluster_e13p21', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label_ratio', type=float, default=1.0,
                        help='Fraction of spots whose RNA/ATAC labels are used in supervision (default 1.0 = full sup)')
    parser.add_argument('--label_seed', type=int, default=0,
                        help='Seed for sampling the labeled spot subset')
    args = parser.parse_args()

    if args.joint_cluster and args.joint_cluster_e13p21:
        parser.error('--joint_cluster and --joint_cluster_e13p21 are mutually exclusive')
    cluster_mode = ('joint_e13p21' if args.joint_cluster_e13p21
                    else 'joint_all' if args.joint_cluster
                    else 'per_time')

    if args.output_dir is None:
        tag = f"r{args.label_ratio}"
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), f'output_semi_{tag}')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("Semi-Supervised Spatial Heterogeneous Graph Encoder")
    print("=" * 60)
    print(f"  hidden={args.hidden}, layers={args.n_layers}, heads={args.num_heads}")
    print(f"  epochs={args.epochs}, lr={args.lr}, amp={args.amp}")
    print(f"  λ: recon={args.lambda_recon} align={args.lambda_align} "
          f"spatial={args.lambda_spatial} gp={args.lambda_gp} "
          f"rna={args.lambda_rna} atac={args.lambda_atac}")
    print(f"  svd={not args.no_svd} dim={args.svd_dim}")
    print(f"  cluster_mode={cluster_mode}")
    print(f"  label_ratio={args.label_ratio} label_seed={args.label_seed}")
    print(f"  [Joint_clusters HELD OUT for evaluation]")

    print("\n" + "-" * 40)
    data = load_combined_data(
        args.data_dir, device=device,
        use_svd=not args.no_svd, svd_dim=args.svd_dim)

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

    trainer = SemiSupervisedTrainer(
        model=model, data=data, device=device,
        lr=args.lr, wd=args.wd,
        lambda_recon=args.lambda_recon, lambda_align=args.lambda_align,
        lambda_spatial=args.lambda_spatial, lambda_gp=args.lambda_gp,
        lambda_rna=args.lambda_rna, lambda_atac=args.lambda_atac,
        temperature=args.temperature, use_amp=args.amp,
        cluster_mode=cluster_mode,
        label_ratio=args.label_ratio, label_seed=args.label_seed,
    )

    print("\n" + "=" * 60)
    history = trainer.train(epochs=args.epochs, eval_every=5)

    trainer.save(os.path.join(args.output_dir, 'model.pt'))

    print("\nExtracting embeddings...")
    emb = trainer.get_embeddings()
    np.save(os.path.join(args.output_dir, 'spot_emb.npy'), emb['spot'])
    np.save(os.path.join(args.output_dir, 'gene_emb.npy'), emb['gene'])
    np.save(os.path.join(args.output_dir, 'peak_emb.npy'), emb['peak'])
    print(f"  spot  {emb['spot'].shape}")
    print(f"  gene  {emb['gene'].shape}")
    print(f"  peak  {emb['peak'].shape}")

    print("\nComputing Gene-Peak scores...")
    gp_scores = trainer.compute_gp_scores()
    np.save(os.path.join(args.output_dir, 'gp_scores.npy'), gp_scores)
    print(f"  gp_scores {gp_scores.shape}")

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
