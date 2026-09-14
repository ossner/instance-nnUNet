import torch
import torch.nn as nn
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


class VectorizedBlobLoss(nn.Module):
    """
    Vectorized Instance-aware Blob Loss (Kofler et al.)
    Computes per-instance Dice loss using GPU scatter_add_ operations.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, net_output: torch.Tensor, instance_target: torch.Tensor) -> torch.Tensor:
        # Standardize foreground probability extraction
        if net_output.shape[1] == 2:
            probs_fg = torch.softmax(net_output, dim=1)[:, 1]
        else:
            probs_fg = torch.sigmoid(net_output)[:, 0]

        B = net_output.shape[0]
        
        # Flatten spatial dimensions -> Shape: [B, N_pixels]
        p_fg_flat = probs_fg.view(B, -1)
        t_inst_flat = instance_target[:, 0].long().view(B, -1)

        # Allocate dynamic buffers on GPU
        max_id = int(t_inst_flat.max().item()) + 1
        if max_id <= 1:  # Only background present
            return (net_output * 0).sum()

        device = net_output.device
        dtype = p_fg_flat.dtype

        intersection = torch.zeros((B, max_id), device=device, dtype=dtype)
        instance_sizes = torch.zeros((B, max_id), device=device, dtype=dtype)

        # Massively parallel group-by sums
        intersection.scatter_add_(1, t_inst_flat, p_fg_flat)
        instance_sizes.scatter_add_(1, t_inst_flat, torch.ones_like(p_fg_flat))

        # Exclude Background (Instance ID 0) -> Shape: [B, max_id - 1]
        valid_mask = instance_sizes[:, 1:] > 0
        
        # Compute individual instance Dice
        cardinality = intersection[:, 1:] + instance_sizes[:, 1:]
        blob_dice = (2.0 * intersection[:, 1:] + self.eps) / (cardinality + self.eps)
        blob_loss = 1.0 - blob_dice

        total_valid_blobs = valid_mask.sum()
        if total_valid_blobs == 0:
            return (net_output * 0).sum()

        return (blob_loss * valid_mask).sum() / total_valid_blobs


class DC_CE_and_Blob_Loss(nn.Module):
    """
    Compound Loss: Standard DC + CE Loss (Binary) + Instance Blob Loss.
    """
    def __init__(
        self, 
        soft_dice_kwargs: dict, 
        ce_kwargs: dict, 
        weight_global: float = 1.0, 
        weight_blob: float = 1.0
    ):
        super().__init__()
        self.global_loss = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, weight_ce=1.0, weight_dice=1.0)
        self.blob_loss = VectorizedBlobLoss()
        
        # Normalize weights so they sum to 1.0
        total_weight = weight_global + weight_blob
        self.weight_global = weight_global / total_weight
        self.weight_blob = weight_blob / total_weight

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        target[:, 0:1] -> Binary target
        target[:, 1:2] -> Instance ID
        target[:, 2:3] -> Voronoi target (Unused here, but retained for indexing consistency)
        """
        binary_target = target[:, 0:1]
        instance_target = target[:, 1:2]

        l_global = self.global_loss(net_output, binary_target)
        l_blob = self.blob_loss(net_output, instance_target)
        return (self.weight_global * l_global) + (self.weight_blob * l_blob)


class nnUNetTrainerBlob(nnUNetTrainer):
    """
    nnUNetv2 Trainer integrating vectorized Blob Loss.
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
            weight_global=1.0,
            weight_blob=1.0
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def train_step(self, batch: dict) -> dict:
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True):
                output = self.network(data)
                del data
                l = self.loss(output, target)

        output_eval = output[0] if isinstance(output, (list, tuple)) else output
        target_eval = target[0] if isinstance(output, (list, tuple)) else target

        binary_target_eval = target_eval[:, 0:1]

        # Standardize evaluation metrics extraction
        axes = tuple(range(2, output_eval.ndim))
        
        probs = torch.softmax(output_eval, dim=1)
        # Take just the foreground class for metric validation
        probs = probs[:, 1:2]

        tp, fp, fn, _ = get_tp_fp_fn_tn(probs, binary_target_eval, axes=axes)

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        }