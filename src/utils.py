import os
import json
import imageio

import torch
import numpy as np

import monai
import monai.metrics

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def psnr(y_pred, y, max_val=1.0, **kwargs) -> torch.Tensor:
    m = monai.metrics.PSNRMetric(max_val=max_val, **kwargs)
    return m(y_pred, y)


def ssim(y_pred, y, data_range=1.0, **kwargs) -> torch.Tensor:
    m = monai.metrics.SSIMMetric(spatial_dims=3 if y.dim() == 5 else 2, data_range=data_range, **kwargs)
    return m(y_pred, y)


def log_to_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def get_middle_slice(volume):

    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu()

    if volume.ndim == 5:
        volume = volume[0, 0]

    D = volume.shape[0]

    return volume[D // 2].numpy()


def save_checkpoint_slices(
    output_path,
    target,
    moving,
    warped
):

    target_slice = get_middle_slice(target)
    moving_slice = get_middle_slice(moving)
    warped_slice = get_middle_slice(warped)

    diff = np.abs(target_slice - warped_slice)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    ax[0].imshow(target_slice, cmap="gray")
    ax[0].set_title("Target")

    ax[1].imshow(moving_slice, cmap="gray")
    ax[1].set_title("Moving")

    ax[2].imshow(warped_slice, cmap="gray")
    ax[2].set_title("Warped")

    ax[3].imshow(diff, cmap="hot")
    ax[3].set_title("Abs Error")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_dvf_orthogonal_slices_png(
    dvf, 
    output_path, 
    stride=16, 
    scale=0.5, 
    cmap="viridis"
):

    dvf = dvf.detach().cpu()
    if dvf.ndim == 5:  # [B, 3, Z, Y, X]
        dvf = dvf[0]

    z_dim, y_dim, x_dim = dvf.shape[1], dvf.shape[2], dvf.shape[3]
    z_mid, y_mid, x_mid = z_dim // 2, y_dim // 2, x_dim // 2

    magnitude_3d = torch.norm(dvf, dim=0)
    max_disp_3d = magnitude_3d.max().item()
    mean_disp_3d = magnitude_3d.mean().item()

    max_val = max(abs(dvf).max().item(), 1e-5)

    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    
    views_config = [
        {
            "name": "Axial (Z-Mid)",
            "slice_idx": z_mid,
            "u_horiz": dvf[2, z_mid].numpy(),  
            "u_vert": dvf[1, z_mid].numpy(),   
            "u_ortho": dvf[0, z_mid].numpy(), 
            "mag": magnitude_3d[z_mid].numpy(),
            "labels": ("Δx (Left-Right)", "Δy (Posterior-Anterior)", "Δz (Inferior-Superior)")
        },
        {
            "name": "Coronal (Y-Mid)",
            "slice_idx": y_mid,
            "u_horiz": dvf[2, :, y_mid].numpy(), 
            "u_vert": dvf[0, :, y_mid].numpy(),  
            "u_ortho": dvf[1, :, y_mid].numpy(), 
            "mag": magnitude_3d[:, y_mid].numpy(),
            "labels": ("Δx (Left-Right)", "Δz (Inferior-Superior)", "Δy (Posterior-Anterior)")
        },
        {
            "name": "Sagittal (X-Mid)",
            "slice_idx": x_mid,
            "u_horiz": dvf[1, :, :, x_mid].numpy(), 
            "u_vert": dvf[0, :, :, x_mid].numpy(),  
            "u_ortho": dvf[2, :, :, x_mid].numpy(), 
            "mag": magnitude_3d[:, :, x_mid].numpy(),
            "labels": ("Δy (Posterior-Anterior)", "Δz (Inferior-Superior)", "Δx (Left-Right)")
        }
    ]

    for row_idx, view in enumerate(views_config):
        ax_row = axes[row_idx]

        uh, uv, uo, mag_slice = view["u_horiz"], view["u_vert"], view["u_ortho"], view["mag"]
        lbl_h, lbl_v, lbl_o = view["labels"]

        im0 = ax_row[0].imshow(uh, cmap="coolwarm", vmin=-max_val, vmax=max_val)
        ax_row[0].set_title(f"{view['name']} - {lbl_h}")
        fig.colorbar(im0, ax=ax_row[0], fraction=0.046, pad=0.04)

        im1 = ax_row[1].imshow(uv, cmap="coolwarm", vmin=-max_val, vmax=max_val)
        ax_row[1].set_title(f"{view['name']} - {lbl_v}")
        fig.colorbar(im1, ax=ax_row[1], fraction=0.046, pad=0.04)

        im2 = ax_row[2].imshow(uo, cmap="coolwarm", vmin=-max_val, vmax=max_val)
        ax_row[2].set_title(f"{view['name']} - {lbl_o}")
        fig.colorbar(im2, ax=ax_row[2], fraction=0.046, pad=0.04)

        im3 = ax_row[3].imshow(mag_slice, cmap=cmap, vmin=0, vmax=max(mag_slice.max(), 1e-5))
        ax_row[3].set_title(f"{view['name']} - Magnitude & Vectors")
        fig.colorbar(im3, ax=ax_row[3], fraction=0.046, pad=0.04)

        nr, nc = mag_slice.shape
        rr = np.arange(0, nr, stride)
        cc = np.arange(0, nc, stride)

        gg_r, gg_c = np.meshgrid(rr, cc, indexing="ij")

        u_s = uh[np.ix_(rr, cc)]
        v_s = uv[np.ix_(rr, cc)]
        
        ax_row[3].quiver(
            gg_c,
            gg_r,
            u_s,
            -v_s,  
            color="white", 
            alpha=0.85,
            scale=scale,  
            scale_units="xy",
            angles="xy",
            width=0.004,
            headwidth=3,
            headlength=4
        )

        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Multi-Planar Deformation Vector Field (DVF) Analysis\n"
        f"Global 3D Max Displacement: {max_disp_3d:.2f} px | "
        f"Global 3D Mean Displacement: {mean_disp_3d:.2f} px",
        fontsize=16, y=0.98
    )

    try:
        plt.tight_layout()
    except Exception:
        pass
    
    out_p = Path(output_path)
    os.makedirs(out_p.parent, exist_ok=True)
    plt.savefig(out_p, dpi=100, bbox_inches="tight")
    plt.close()


def save_optimization_frame(
    output_path,
    moving,
    target,
    warped,
    loss_history,
    iteration=None,
    total_iterations=None,
    psnr_val=None,
    ssim_val=None,
):
    moving_slice = get_middle_slice(moving)
    target_slice = get_middle_slice(target)
    warped_slice = get_middle_slice(warped)
    error = target_slice - warped_slice
    if not hasattr(save_optimization_frame, "static_vmax"):
        calculated_vmax = np.percentile(np.abs(error), 99)
        save_optimization_frame.static_vmax = max(calculated_vmax, 1e-8)
    vmax = save_optimization_frame.static_vmax
    current_loss = loss_history[-1]
    fig = plt.figure(figsize=(15, 3.8))
    gs_master = gridspec.GridSpec(1, 2, width_ratios=[4, 1.2], wspace=0.3)
    gs_images = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs_master[0], wspace=0.02
    )
    ax = []
    for i in range(4):
        ax.append(fig.add_subplot(gs_images[i]))
    ax.append(fig.add_subplot(gs_master[1]))
    ax[0].imshow(moving_slice, cmap="gray")
    ax[0].set_title("Moving", fontsize=10)
    ax[1].imshow(target_slice, cmap="gray")
    ax[1].set_title("Target", fontsize=10)
    ax[2].imshow(warped_slice, cmap="gray")
    ax[2].set_title("Warped", fontsize=10)
    im = ax[3].imshow(error, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[3].set_title("Target - Warped", fontsize=10)
    cbar = fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    ax[4].plot(loss_history, linewidth=2, color="tab:blue")
    ax[4].set_xlabel("Iteration", fontsize=8)
    ax[4].set_ylabel("Loss", fontsize=8)
    ax[4].tick_params(axis="both", labelsize=8)
    if total_iterations:
        ax[4].set_xlim(0, total_iterations)
    ax[4].grid(True, alpha=0.3)
    ax[4].scatter(len(loss_history) - 1, current_loss, color="tab:red", s=25, zorder=5)

    loss_label = f"Loss = {current_loss:.6f}"
    if psnr_val is not None:
        loss_label += f"  |  PSNR = {psnr_val:.2f}"
    if ssim_val is not None:
        loss_label += f"  |  SSIM = {ssim_val:.4f}"
    ax[4].set_title(loss_label, fontsize=8)

    for i in range(4):
        ax[i].axis("off")
    if iteration is not None:
        fig.suptitle(
            f"Iteration {iteration} | Loss = {current_loss:.6f}"
            + (f" | PSNR = {psnr_val:.2f}" if psnr_val is not None else "")
            + (f" | SSIM = {ssim_val:.4f}" if ssim_val is not None else ""),
            fontsize=11,
            fontweight="bold",
            y=0.98,
        )
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    

def create_optimization_gif(
    frames_dir,
    output_gif,
    fps=5
):
    frame_files = sorted(
        [
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.endswith(".png")
        ]
    )

    images = [
        imageio.imread(f)
        for f in frame_files
    ]

    H = max(im.shape[0] for im in images)
    W = max(im.shape[1] for im in images)

    padded = []

    for im in images:

        canvas = np.ones(
            (H, W, 4),
            dtype=np.uint8
        ) * 255

        canvas[
            :im.shape[0],
            :im.shape[1]
        ] = im

        padded.append(canvas)

    imageio.mimsave(
        output_gif,
        padded,
        fps=fps,
        loop=0,  
    )