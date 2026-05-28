"""
Semi-Supervised Trainer
=======================
Same loss formulation as SupervisedTrainer, but only a fraction `label_ratio`
of spots contribute to L_rna / L_atac. The remaining spots are masked to -1
so the parent trainer's existing `>= 0` filter drops them automatically.

The encoder, loss functions, and training loop are reused from supervised.model
without modification.
"""

import torch

from supervised.model import SupervisedEncoder, SupervisedTrainer  # noqa: F401


class SemiSupervisedTrainer(SupervisedTrainer):
    def __init__(self, model, data, label_ratio=1.0, label_seed=0, **kwargs):
        if not (0.0 < label_ratio <= 1.0):
            raise ValueError(f"label_ratio must be in (0, 1], got {label_ratio}")

        data = dict(data)
        if label_ratio < 1.0:
            n_spots = data['n_spots']
            dev = data['rna_labels'].device
            gen = torch.Generator(device='cpu').manual_seed(label_seed)
            keep = (torch.rand(n_spots, generator=gen) < label_ratio).to(dev)

            rna = data['rna_labels'].clone()
            atac = data['atac_labels'].clone()
            rna_orig = (rna >= 0).sum().item()
            atac_orig = (atac >= 0).sum().item()
            rna[~keep] = -1
            atac[~keep] = -1
            rna_kept = (rna >= 0).sum().item()
            atac_kept = (atac >= 0).sum().item()
            data['rna_labels'] = rna
            data['atac_labels'] = atac

            print(f"[semi-sup] label_ratio={label_ratio}, seed={label_seed}")
            print(f"          RNA labels kept  {rna_kept}/{rna_orig}")
            print(f"          ATAC labels kept {atac_kept}/{atac_orig}")
        else:
            print(f"[semi-sup] label_ratio=1.0 (identical to full supervised)")

        self.label_ratio = label_ratio
        self.label_seed = label_seed
        super().__init__(model, data, **kwargs)
