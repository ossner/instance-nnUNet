import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerDebug(nnUNetTrainer):
    """Debug trainer to validate 3-channel (Binary, Instance, Voronoi) targets

    at full resolution and lower deep supervision scales.
    """
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        
    def train_step(self, batch: dict) -> dict:
        # Check if we haven't rendered for the current epoch yet
        # if getattr(self, '_last_debug_epoch', -1) != self.current_epoch:
        #     self._last_debug_epoch = self.current_epoch

        #     # 1. Run evaluation forward pass for prediction plot
        #     self.network.eval()
        #     with torch.no_grad():
        #         data = batch['data']
        #         if not isinstance(data, torch.Tensor):
        #             data = torch.from_numpy(data)
        #         data = data.to(self.device, non_blocking=True)
                
        #         outputs = self.network(data)  # Deep supervision output list
        #     self.network.train()

        #     # 2. Render and save targets vs prediction
        #     visualize_epoch_debug(batch['target'], outputs, epoch=self.current_epoch)
        #     print(f"[DEBUG] Saved epoch {self.current_epoch} prediction visualization.")

        # Strip targets for normal loss calculation
        # Check values in object foreground region vs background region
        batch['target'] = [t[:, 0:1] for t in batch['target']]
        return super().train_step(batch)
    
    def validation_step(self, batch: dict) -> dict:
        batch['target'] = [t[:, 0:1] for t in batch['target']]
        return super().validation_step(batch)
    

def visualize_epoch_debug(targets: list, outputs: list, epoch: int, sample_idx: int = 0):
    num_levels = len(targets)
    # 4 Columns: 3 Targets + 1 Prediction
    fig, axes = plt.subplots(num_levels, 4, figsize=(16, 3 * num_levels))
    
    headers = ["Target: Binary", "Target: Instance", "Target: Voronoi", "Prediction (Sigmoid)"]
    cmaps = ['gray', 'nipy_spectral', 'viridis', 'gray']

    for level_idx, (target_tensor, out_tensor) in enumerate(zip(targets, outputs)):
        tgt_sample = target_tensor[sample_idx].detach().cpu().numpy()
        # Sigmoid on output logits for channel 0
        pred_sample = torch.sigmoid(out_tensor[sample_idx, 0]).detach().cpu().numpy()
        if tgt_sample.ndim == 3: # input is 2D
            maps = [tgt_sample[0], tgt_sample[1], tgt_sample[2], pred_sample] # Binary target, Instance Map, Voronoi Map, Preds
        elif tgt_sample.ndim == 4: # input is 3D
            maps = [tgt_sample[0][tgt_sample.shape[1]//2], tgt_sample[1][tgt_sample.shape[1]//2], tgt_sample[2][tgt_sample.shape[1]//2], pred_sample[tgt_sample.shape[1]//2]] # Binary target, Instance Map, Voronoi Map, Preds
        else:
            raise NotImplementedError("Input data seems to be neither 2D nor 3D")

        for col_idx in range(4):
            ax = axes[level_idx, col_idx] if num_levels > 1 else axes[col_idx]
            im = ax.imshow(maps[col_idx], cmap=cmaps[col_idx], interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if level_idx == 0:
                ax.set_title(headers[col_idx], fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"L{level_idx} ({tgt_sample.shape[1]}x{tgt_sample.shape[2]})", fontsize=11, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"debug_epoch_{epoch:03d}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)