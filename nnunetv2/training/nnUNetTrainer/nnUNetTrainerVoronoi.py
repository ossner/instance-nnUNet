import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


class VoronoiRegionLoss(nn.Module):
    """
    Computes region-wise Dice and Cross-Entropy loss based on a Voronoi mask.
    Fully vectorized using scatter_add_ for massive GPU acceleration.
    Excludes Region 0 (nnUNet padding/background out of bounds).
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, net_output: torch.Tensor, binary_target: torch.Tensor, voronoi_target: torch.Tensor) -> torch.Tensor:
        B = net_output.shape[0]
        
        # 1. Compute global unreduced CE and extract Foreground probabilities
        probs = torch.softmax(net_output, dim=1)
        probs_fg = probs[:, 1]
        ce_loss_unreduced = F.cross_entropy(
            net_output, binary_target[:, 0].long(), reduction='none'
        )

        # 2. Flatten spatial dimensions -> Shape: [B, N_pixels]
        p_fg_flat = probs_fg.view(B, -1)
        t_bin_flat = binary_target[:, 0].float().view(B, -1)
        t_vor_flat = voronoi_target[:, 0].long().view(B, -1)
        ce_unred_flat = ce_loss_unreduced.view(B, -1)

        # 3. Find global max Voronoi ID to allocate accumulation buffers
        max_id = int(t_vor_flat.max().item()) + 1
        
        # Failsafe: if batch only contains background (0)
        if max_id == 1:
            return (net_output * 0).sum()

        # 4. Initialize GPU accumulators (Shape: [B, max_id])
        device = net_output.device
        dtype = p_fg_flat.dtype
        
        region_counts = torch.zeros((B, max_id), device=device, dtype=dtype)
        ce_sum        = torch.zeros((B, max_id), device=device, dtype=dtype)
        intersection  = torch.zeros((B, max_id), device=device, dtype=dtype)
        sum_p         = torch.zeros((B, max_id), device=device, dtype=dtype)
        sum_t         = torch.zeros((B, max_id), device=device, dtype=dtype)

        # 5. Scatter Add (Massively parallel Group-By operations)
        region_counts.scatter_add_(1, t_vor_flat, torch.ones_like(p_fg_flat))
        ce_sum.scatter_add_(1, t_vor_flat, ce_unred_flat)
        intersection.scatter_add_(1, t_vor_flat, p_fg_flat * t_bin_flat)
        sum_p.scatter_add_(1, t_vor_flat, p_fg_flat)
        sum_t.scatter_add_(1, t_vor_flat, t_bin_flat)

        # 6. Exclude Region 0 (Slice out the first column)
        # Tensors become shape: [B, max_id - 1]
        valid_mask = region_counts[:, 1:] > 0
        
        # Region CE: Sum of CE / Pixel count
        # (add 1e-8 to prevent division by zero in empty regions; valid_mask ignores them anyway)
        region_ce = ce_sum[:, 1:] / (region_counts[:, 1:] + 1e-8)
        
        # Region Dice
        cardinality = sum_p[:, 1:] + sum_t[:, 1:]
        region_dice = 1.0 - (2.0 * intersection[:, 1:] + self.eps) / (cardinality + self.eps)

        # Combine
        region_loss = region_ce + region_dice

        # 7. Final reduction
        total_valid_regions = valid_mask.sum()
        
        # Failsafe for deep supervision heavily downsampled patches
        if total_valid_regions == 0:
            return (net_output * 0).sum()

        # Sum only the valid regions and average
        return (region_loss * valid_mask).sum() / total_valid_regions


class DC_CE_and_Voronoi_Loss(nn.Module):
    """
    Compound Loss: 
    Global DC + CE Loss (Standard nnUNet) + Region-wise DC + CE (Voronoi Mask)
    """
    def __init__(
        self, 
        soft_dice_kwargs: dict, 
        ce_kwargs: dict, 
        weight_global: float = 1, 
        weight_voronoi: float = 1
    ):
        super().__init__()
        # Standard global loss calculated over the padded image
        self.global_loss = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, weight_ce=1.0, weight_dice=1.0)
        self.voronoi_loss = VoronoiRegionLoss()
        
        self.weight_global = weight_global
        self.weight_voronoi = weight_voronoi

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        target[:, 0:1] -> Binary target
        target[:, 1:2] -> Instance ID (Unused here, but retained for indexing consistency)
        target[:, 2:3] -> Voronoi target
        """
        binary_target = target[:, 0:1]
        voronoi_target = target[:, 2:3]

        l_global = self.global_loss(net_output, binary_target)
        l_voronoi = self.voronoi_loss(net_output, binary_target, voronoi_target)

        return (self.weight_global * l_global) + (self.weight_voronoi * l_voronoi)


class nnUNetTrainerVoronoi(nnUNetTrainer):
    """
    Custom nnUNetv2 Trainer integrating Voronoi region-wise Dice+CE loss.
    """
    def _build_loss(self):
        loss = DC_CE_and_Voronoi_Loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp  # Prevents single-GPU DDP crashes
            },
            ce_kwargs={
                'ignore_index': self.label_manager.ignore_label if self.label_manager.ignore_label is not None else -100
            },
            weight_global=0.5,
            weight_voronoi=0.5  # Adjust weights here as desired
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def train_step(self, batch: dict) -> dict:
        # Feed the multi-channel target as-is to self.loss
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        # TODO: This prevents a strange metric tracking crash, should be investigated further
        # 1. Forward pass & multi-channel loss evaluation
        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True):
                output = self.network(data)
                del data
                l = self.loss(output, target)

        # 2. Extract full-resolution (Scale 0) outputs for metrics
        output_eval = output[0] if isinstance(output, (list, tuple)) else output
        target_eval = target[0] if isinstance(target, (list, tuple)) else target

        # 3. Prevent metric tracking crash (CUDA assert) by slicing down to binary mask
        binary_target_eval = target_eval[:, 0:1]

        # 4. Standard validation metric tracking
        axes = tuple(range(2, output_eval.ndim))
        
        probs = torch.softmax(output_eval, dim=1)
        # Take just the foreground class for metric validation
        probs = probs[:, 1:2] 

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