import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

class AdaptiveVoronoiRegionLoss(nn.Module):
    """
    Computes region-wise Adaptive Tversky and Cross-Entropy loss based on a Voronoi mask.
    Fully vectorized using scatter_add_ for GPU acceleration.
    Adaptive scaling dynamically weights alpha/beta based on local FP/FN fractions.
    """
    def __init__(self, eps: float = 1e-5, A: float = 0.3, B: float = 0.4):
        super().__init__()
        self.eps = eps
        self.A = A
        self.B = B

    def forward(self, net_output: torch.Tensor, binary_target: torch.Tensor, voronoi_target: torch.Tensor) -> torch.Tensor:
        batch_size = net_output.shape[0]

        # 1. Compute global unreduced CE and extract Foreground probabilities
        probs = torch.softmax(net_output, dim=1)
        probs_fg = probs[:, 1]
        ce_loss_unreduced = F.cross_entropy(
            net_output, binary_target[:, 0].long(), reduction='none'
        )

        # 2. Flatten spatial dimensions -> Shape: [B, N_pixels]
        p_fg_flat = probs_fg.view(batch_size, -1)
        t_bin_flat = binary_target[:, 0].float().view(batch_size, -1)
        t_vor_flat = voronoi_target[:, 0].long().view(batch_size, -1)
        ce_unred_flat = ce_loss_unreduced.view(batch_size, -1)

        # 3. Find global max Voronoi ID to allocate accumulation buffers
        max_id = int(t_vor_flat.max().item()) + 1

        if max_id == 1:
            return (net_output * 0).sum()

        # 4. Initialize GPU accumulators (Shape: [B, max_id])
        device = net_output.device
        dtype = p_fg_flat.dtype

        region_counts = torch.zeros((batch_size, max_id), device=device, dtype=dtype)
        ce_sum        = torch.zeros((batch_size, max_id), device=device, dtype=dtype)
        tp            = torch.zeros((batch_size, max_id), device=device, dtype=dtype)
        sum_p         = torch.zeros((batch_size, max_id), device=device, dtype=dtype)
        sum_t         = torch.zeros((batch_size, max_id), device=device, dtype=dtype)

        # 5. Scatter Add (Optimized: scatter TP, sum_p, sum_t to avoid large FP/FN spatial tensors)
        region_counts.scatter_add_(1, t_vor_flat, torch.ones_like(p_fg_flat))
        ce_sum.scatter_add_(1, t_vor_flat, ce_unred_flat)
        tp.scatter_add_(1, t_vor_flat, p_fg_flat * t_bin_flat)
        sum_p.scatter_add_(1, t_vor_flat, p_fg_flat)
        sum_t.scatter_add_(1, t_vor_flat, t_bin_flat)

        # 6. Exclude Region 0 (Slice out padding/background)
        valid_mask = region_counts[:, 1:] > 0

        tp_val = tp[:, 1:]
        # Derived region-level FP and FN
        fp_val = torch.clamp(sum_p[:, 1:] - tp_val, min=0.0)
        fn_val = torch.clamp(sum_t[:, 1:] - tp_val, min=0.0)

        # Region CE
        region_ce = ce_sum[:, 1:] / (region_counts[:, 1:] + 1e-8)

        # 7. Compute Adaptive Alpha and Beta per Region
        fp_fn_sum = fp_val + fn_val
        # If FP + FN == 0 (perfect region), default ratio to 0.5 to keep alpha = beta = 0.5
        fp_ratio = torch.where(fp_fn_sum > 0, fp_val / (fp_fn_sum + 1e-8), 0.5)

        alpha_adapt = self.A + self.B * fp_ratio
        beta_adapt = self.A + self.B * (1.0 - fp_ratio)

        # 8. Compute Region-wise Adaptive Tversky (With Numerator Smoothing)
        tversky_den = tp_val + alpha_adapt * fp_val + beta_adapt * fn_val + self.eps
        region_tversky = (tp_val + self.eps) / tversky_den
        region_tversky_loss = 1.0 - region_tversky

        # Combine
        region_loss = region_ce + region_tversky_loss

        # 9. Final reduction
        total_valid_regions = valid_mask.sum()

        if total_valid_regions == 0:
            return (net_output * 0).sum()

        return (region_loss * valid_mask).sum() / total_valid_regions


class DC_CE_and_AdaptiveVoronoi_Loss(nn.Module):
    """
    Compound Loss: 
    Global DC + CE Loss (Standard nnUNet) + Region-wise Adaptive Tversky + CE (Voronoi Mask)
    """
    def __init__(
        self, 
        soft_dice_kwargs: dict, 
        ce_kwargs: dict, 
        weight_global: float = 1.0, 
        weight_voronoi: float = 1.0,
        A: float = 0.3,
        B: float = 0.4
    ):
        super().__init__()
        self.global_loss = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, weight_ce=1.0, weight_dice=1.0)
        self.voronoi_adaptive_loss = AdaptiveVoronoiRegionLoss(A=A, B=B)
        
        # Restored weight normalization matching baseline
        total_weight = weight_global + weight_voronoi
        self.weight_global = weight_global / total_weight
        self.weight_voronoi = weight_voronoi / total_weight

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        binary_target = target[:, 0:1]
        voronoi_target = target[:, 2:3]

        l_global = self.global_loss(net_output, binary_target)
        l_voronoi = self.voronoi_adaptive_loss(net_output, binary_target, voronoi_target)

        return (self.weight_global * l_global) + (self.weight_voronoi * l_voronoi)

class nnUNetTrainerVoronoiAdaptive(nnUNetTrainer):
    """
    Custom nnUNetv2 Trainer integrating Adaptive Region-Specific Tversky loss based on Voronoi regions.
    """
    def _build_loss(self):
        loss = DC_CE_and_AdaptiveVoronoi_Loss(
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
            weight_voronoi=1.0,
            A=0.3,  # Base constant from Chen et al.
            B=0.4   # Dynamic modifier from Chen et al.
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
        # Same exact highly-optimized validation pipeline as your standard Voronoi implementation
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
            'fn_hard': fn.detach().cpu().numpy()
        }