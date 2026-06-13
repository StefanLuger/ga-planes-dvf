import json
import torch
import numpy as np
import matplotlib.pyplot as plt

def log_to_json(filepath: str, data: dict) -> None:
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def save_checkpoint_slices(filepath: str, target: torch.Tensor, moving: torch.Tensor, warped: torch.Tensor) -> None:
    t_np = target[0, 0].detach().cpu().numpy()
    m_np = moving[0, 0].detach().cpu().numpy()
    w_np = warped[0, 0].detach().cpu().numpy()
    err_np = np.abs(t_np - w_np)

    D, H, W = t_np.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    slices = {
        "Axial": (t_np[mid_d, :, :], m_np[mid_d, :, :], w_np[mid_d, :, :], err_np[mid_d, :, :]),
        "Sagittal": (t_np[:, :, mid_w], m_np[:, :, mid_w], w_np[:, :, mid_w], err_np[:, :, mid_w]),
        "Coronal": (t_np[:, mid_h, :], m_np[:, mid_h, :], w_np[:, mid_h, :], err_np[:, mid_h, :])
    }

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for idx, (orient, (t_sl, m_sl, w_sl, e_sl)) in enumerate(slices.items()):
        axes[idx, 0].imshow(t_sl, cmap='gray')
        axes[idx, 0].set_title(f"{orient} Target")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(m_sl, cmap='gray')
        axes[idx, 1].set_title(f"{orient} Moving")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(w_sl, cmap='gray')
        axes[idx, 2].set_title(f"{orient} Warped")
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(e_sl, cmap='hot')
        axes[idx, 3].set_title(f"{orient} Error")
        axes[idx, 3].axis('off')

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()