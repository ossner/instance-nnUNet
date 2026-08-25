import torch
import torch.nn as nn
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

class BlobLoss(nn.Module):
    """
    Instance-aware Blob Loss (Kofler et al.)
    
    Computes Dice loss individually for each connected component / instance present in 
    the ground truth instance map. This prevents small objects/blobs from being 
    dominated by large background regions during gradient updates.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, net_output: torch.Tensor, instance_target: torch.Tensor) -> torch.Tensor:
        """
        net_output: [B, 1, H, W] or [B, 1, D, H, W] (raw logits)
        instance_target: [B, 1, H, W] or [B, 1, D, H, W] (0=BG, 1..N=Instance IDs)
        """
        probs = torch.sigmoid(net_output)
        
        total_blob_loss = 0.0
        total_blobs = 0
        batch_size = net_output.shape[0]

        for b in range(batch_size):
            p_b = probs[b, 0]
            inst_b = instance_target[b, 0]

            # Identify unique instance IDs for the current sample (excluding background 0)
            unique_insts = torch.unique(inst_b)
            unique_insts = unique_insts[unique_insts > 0]

            if len(unique_insts) == 0:
                continue

            for inst_id in unique_insts:
                mask = (inst_b == inst_id).float()
                
                # Instance-specific dice term
                intersection = torch.sum(p_b * mask)
                cardinality = torch.sum(p_b * mask) + torch.sum(mask)
                
                blob_dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
                total_blob_loss += (1.0 - blob_dice)
                total_blobs += 1

        if total_blobs == 0:
            return torch.tensor(0.0, device=net_output.device, dtype=net_output.dtype, requires_grad=True)

        return total_blob_loss / total_blobs


class DC_CE_and_Blob_Loss(nn.Module):
    """
    Compound Loss: Combines Standard DC + CE Loss (on Channel 0: Binary Target)
    with Blob Loss (on Channel 1: Instance Target).
    """
    def __init__(
        self, 
        soft_dice_kwargs: dict, 
        ce_kwargs: dict, 
        weight_dc_ce: float = 1.0, 
        weight_blob: float = 1.0
    ):
        super().__init__()
        self.dc_ce = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, weight_ce=1.0, weight_dice=1.0)
        self.blob_loss = BlobLoss()
        self.weight_dc_ce = weight_dc_ce
        self.weight_blob = weight_blob

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        target: multi-channel tensor where:
            target[:, 0:1] -> Binary Semantic Target
            target[:, 1:2] -> Instance ID Map
        """
        binary_target = target[:, 0:1]
        instance_target = target[:, 1:2]

        l_dc_ce = self.dc_ce(net_output, binary_target)
        l_blob = self.blob_loss(net_output, instance_target)
        return (self.weight_dc_ce * l_dc_ce) + (self.weight_blob * l_blob)


class nnUNetTrainerBlob(nnUNetTrainer):
    """
    nnUNetv2 Trainer integrating Kofler et al. Blob Loss with Deep Supervision support.
    Handles multi-channel targets safely across training and validation steps.
    """
    def _build_loss(self):
        loss = DC_CE_and_Blob_Loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp
            },
            ce_kwargs={
                'ignore_index': self.label_manager.ignore_label if self.label_manager.ignore_label is not None else -100
            },
            weight_dc_ce=1.0,
            weight_blob=1.0
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def train_step(self, batch: dict) -> dict:
        # Keep multi-channel target intact for loss computation
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # 1. Forward pass & Multi-channel Loss computation
        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True):
                output = self.network(data)
                del data
                l = self.loss(output, target)

        # 2. Extract full-resolution scale (index 0) for metric calculation
        output_eval = output[0] if isinstance(output, (list, tuple)) else output
        target_eval = target[0] if isinstance(target, (list, tuple)) else target

        # 3. Slice target to Channel 0 (Binary target only)
        binary_target_eval = target_eval[:, 0:1]

        # 4. Compute metrics using get_tp_fp_fn_tn from nnunetv2.training.loss.dice
        axes = tuple(range(2, output_eval.ndim))  # Spatial axes (H, W) or (D, H, W)
        probs = torch.sigmoid(output_eval)

        tp, fp, fn, _ = get_tp_fp_fn_tn(probs, binary_target_eval, axes=axes)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard
        }