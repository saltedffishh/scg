"""
Training script for Spatial Heterogeneous Graph Encoder.

Usage:
    python train.py [--epochs 200] [--lr 1e-3] [--hidden 256] [--device cpu]

Loads graph_combined/, trains the encoder, saves model and embeddings.
"""

import argparse
import os
import sys
import numpy as np
import torch

from model import (
    SpatialHeteroEncoder,
    ClusterHeads,
    Trainer,
    load_combined_data,
)


def main():
    parser = argparse.ArgumentParser(description='Train Spatial Hetero Graph Encoder')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graph_combined'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--pos_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--lambda_align', type=float, default=0.1)
    parser.add_argument('--lambda_spatial', type=float, default=0.1)
    parser.add_argument('--lambda_gp', type=float, default=1.0)
    parser.add_argument('--lambda_rna', type=float, default=0.5)
    parser.add_argument('--lambda_atac', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_svd', action='store_true',
                        help='Use raw features instead of SVD reduction')
    parser.add_argument('--svd_dim', type=int, default=100,
                        help='SVD dimension (default 100)')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Spatial Heterogeneous Graph Encoder - Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Hidden dim: {args.hidden}, Layers: {args.n_layers}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")

    # Load data
    print("\n" + "=" * 60)
    print("Loading data...")
    data = load_combined_data(args.data_dir, device=args.device,
                             use_svd=not args.no_svd, svd_dim=args.svd_dim)

    # Create model
    print("\n" + "=" * 60)
    print("Creating model...")
    model = SpatialHeteroEncoder(
        n_spots=data['n_spots'],
        n_genes=data['n_genes'],
        n_peaks=data['n_peaks'],
        rna_dim=data['rna_dim'],
        atac_dim=data['atac_dim'],
        gene_in_dim=data.get('gene_in_dim', data['n_spots']),
        peak_in_dim=data.get('peak_in_dim', data['n_spots']),
        hidden_dim=args.hidden,
        pos_dim=args.pos_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        data=data,
        device=args.device,
        lr=args.lr,
        wd=args.wd,
        lambda_align=args.lambda_align,
        lambda_spatial=args.lambda_spatial,
        lambda_gp=args.lambda_gp,
        lambda_rna=args.lambda_rna,
        lambda_atac=args.lambda_atac,
        temperature=args.temperature,
    )

    # Train
    print("\n" + "=" * 60)
    print("Training...")
    history = trainer.train(epochs=args.epochs, eval_every=5)

    # Save model
    model_path = os.path.join(args.output_dir, 'model.pt')
    trainer.save(model_path)

    # Save embeddings
    print("\nExtracting final embeddings...")
    embeddings = trainer.get_embeddings()

    np.save(os.path.join(args.output_dir, 'spot_emb.npy'), embeddings['spot'])
    np.save(os.path.join(args.output_dir, 'gene_emb.npy'), embeddings['gene'])
    np.save(os.path.join(args.output_dir, 'peak_emb.npy'), embeddings['peak'])
    print(f"Embeddings saved: spot {embeddings['spot'].shape}, "
          f"gene {embeddings['gene'].shape}, peak {embeddings['peak'].shape}")

    # Save training history
    import json
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone! Output saved to {args.output_dir}")


if __name__ == '__main__':
    main()
