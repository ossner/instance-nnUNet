import torch
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class nnUNetTrainerBaseline(nnUNetTrainer):
    """
    Baseline nnUNetv2 Trainer for fair comparison against Blob and Voronoi loss trainers.
    
    It accepts the same multi-channel dataloader outputs but strips the Instance 
    and Voronoi masks, computing ONLY the standard global Dice + Cross-Entropy loss 
    on the Binary target.
    """
    def _build_loss(self):
        # 1. Standard nnUNet Global Dice + CE Loss
        loss = DC_and_CE_loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp  # Retained: Prevents single-GPU DDP crashes
            },
            ce_kwargs={
                'ignore_index': self.label_manager.ignore_label if self.label_manager.ignore_label is not None else -100
            },
            weight_ce=1.0,
            weight_dice=1.0
        )

        # 2. Deep supervision wrapper
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def train_step(self, batch: dict) -> dict:
        # Strip targets down to Channel 0 (Binary Semantic Mask)
        # Deep supervision provides targets at multiple resolution scales
        if isinstance(batch['target'], list):
            batch['target'] = [t[:, 0:1] for t in batch['target']]
        else:
            batch['target'] = batch['target'][:, 0:1]
            
        # Call standard nnUNet train step with purely binary targets
        return super().train_step(batch)
    
    def validation_step(self, batch: dict) -> dict:
        # Strip targets down to Channel 0 (Binary Semantic Mask)
        if isinstance(batch['target'], list):
            batch['target'] = [t[:, 0:1] for t in batch['target']]
        else:
            batch['target'] = batch['target'][:, 0:1]
        return super().validation_step(batch)