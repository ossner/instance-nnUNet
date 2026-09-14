import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


class InstanceBalancedVoronoiLoss(nn.Module):
    """
    Computes balanced region-wise Dice and Cross-Entropy loss.
    Assumes 1 Voronoi Region = 1 Instance.
    
    1. Intra-Region: Decouples Instance (FG) and Local Background (BG) CE to prevent dilution.
    2. Inter-Region: Weights region losses by sqrt(Instance Volume) to balance 
       large (easy) and small (volatile) instances, keeping global loss magnitude stable.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, net_output: torch.Tensor, binary_target: torch.Tensor, voronoi_target: torch.Tensor) -> torch.Tensor:
        B = net_output.shape[0]
        device = net_output.device
        
        # 1. Global Probs and Unreduced CE
        probs = torch.softmax(net_output, dim=1)
        probs_fg = probs[:, 1]
        
        ce_unreduced = F.cross_entropy(
            net_output, binary_target[:, 0].long(), reduction='none'
        )

        target_fg = binary_target[:, 0].float()
        target_bg = 1.0 - target_fg

        # 2. Flatten spatial dimensions -> Shape: [B, N_pixels]
        p_fg_flat  = probs_fg.view(B, -1)
        t_fg_flat  = target_fg.view(B, -1)
        t_bg_flat  = target_bg.view(B, -1)
        t_vor_flat = voronoi_target[:, 0].long().view(B, -1)
        
        ce_fg_flat = (ce_unreduced * target_fg).view(B, -1)
        ce_bg_flat = (ce_unreduced * target_bg).view(B, -1)

        max_id = int(t_vor_flat.max().item()) + 1
        if max_id == 1:
            return (net_output * 0).sum()

        dtype = p_fg_flat.dtype

        # 3. Initialize GPU Accumulators
        v_fg         = torch.zeros((B, max_id), device=device, dtype=dtype)
        v_bg         = torch.zeros((B, max_id), device=device, dtype=dtype)
        ce_fg_sum    = torch.zeros((B, max_id), device=device, dtype=dtype)
        ce_bg_sum    = torch.zeros((B, max_id), device=device, dtype=dtype)
        intersection = torch.zeros((B, max_id), device=device, dtype=dtype)
        sum_p        = torch.zeros((B, max_id), device=device, dtype=dtype)

        # 4. Massively parallel Group-By operations
        v_fg.scatter_add_(1, t_vor_flat, t_fg_flat)
        v_bg.scatter_add_(1, t_vor_flat, t_bg_flat)
        ce_fg_sum.scatter_add_(1, t_vor_flat, ce_fg_flat)
        ce_bg_sum.scatter_add_(1, t_vor_flat, ce_bg_flat)
        intersection.scatter_add_(1, t_vor_flat, p_fg_flat * t_fg_flat)
        sum_p.scatter_add_(1, t_vor_flat, p_fg_flat)

        # 5. Exclude Region 0 (Padding/Global Background)
        v_fg_valid = v_fg[:, 1:]
        v_bg_valid = v_bg[:, 1:]
        ce_fg_valid = ce_fg_sum[:, 1:]
        ce_bg_valid = ce_bg_sum[:, 1:]
        inter_valid = intersection[:, 1:]
        sum_p_valid = sum_p[:, 1:]

        # valid_mask finds regions present in THIS specific cropped patch
        valid_mask = (v_fg_valid + v_bg_valid) > 0

        # --- Intra-Region Class-Balanced CE ---
        # Normalize FG and BG separately to prevent large regions from diluting the instance
        ce_fg_norm = ce_fg_valid / (v_fg_valid + self.eps)
        ce_bg_norm = ce_bg_valid / (v_bg_valid + self.eps)

        # Handle patch-cropping edge cases gracefully (if patch only caught BG or FG of a region)
        weight_fg = (v_fg_valid > 0).float()
        weight_bg = (v_bg_valid > 0).float()
        weight_sum = weight_fg + weight_bg + self.eps

        # Averages the components present (usually 0.5 * FG_CE + 0.5 * BG_CE)
        region_ce = (ce_fg_norm * weight_fg + ce_bg_norm * weight_bg) / weight_sum

        # --- Region Soft Dice ---
        # Dice inherently focuses purely on the instance, so local imbalance isn't an issue here
        cardinality = sum_p_valid + v_fg_valid 
        region_dice = 1.0 - (2.0 * inter_valid + self.eps) / (cardinality + self.eps)

        region_loss = region_ce + region_dice

        # --- Inter-Region Instance-Scale Weighting ---
        # Weight by the square root of the instance volume.
        # This prevents tiny noisy instances from generating identical gradients to massive obvious lesions,
        # smoothing out the loss landscape while maintaining strong focus on multi-instance separation.
        inter_region_weights = torch.sqrt(v_fg_valid + self.eps) * valid_mask.float()
        total_weight = inter_region_weights.sum()

        if total_weight == 0:
            return (net_output * 0).sum()

        # Weighted loss sum, normalized by total weight to keep final magnitude strictly consistent
        return (region_loss * inter_region_weights).sum() / total_weight


class DC_CE_and_InstanceBalancedVoronoi_Loss(nn.Module):
    """
    Compound Loss: Global DC + CE Loss + Instance-Balanced Region-wise DC + CE
    """
    def __init__(
        self, 
        soft_dice_kwargs: dict, 
        ce_kwargs: dict, 
        weight_global: float = 1.0, 
        weight_voronoi: float = 1.0
    ):
        super().__init__()
        self.global_loss = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, weight_ce=1.0, weight_dice=1.0)
        self.voronoi_loss = InstanceBalancedVoronoiLoss()
        
        total_weight = weight_global + weight_voronoi
        self.weight_global = weight_global / total_weight
        self.weight_voronoi = weight_voronoi / total_weight

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        binary_target = target[:, 0:1]
        voronoi_target = target[:, 2:3]

        l_global = self.global_loss(net_output, binary_target)
        l_voronoi = self.voronoi_loss(net_output, binary_target, voronoi_target)

        return (self.weight_global * l_global) + (self.weight_voronoi * l_voronoi)


class nnUNetTrainerBalancedVoronoi(nnUNetTrainer):
    """
    Custom nnUNetv2 Trainer integrating Instance-Balanced Voronoi Region Loss.
    """
    def _build_loss(self):
        loss = DC_CE_and_InstanceBalancedVoronoi_Loss(
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
            weight_voronoi=1.0 
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
        target_eval = target[0] if isinstance(target, (list, tuple)) else target
        binary_target_eval = target_eval[:, 0:1]

        axes = tuple(range(2, output_eval.ndim))
        probs = torch.softmax(output_eval, dim=1)[:, 1:2]
        
        tp, fp, fn, _ = get_tp_fp_fn_tn(probs, binary_target_eval, axes=axes)

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy(),
        }