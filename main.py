import os
from hydra.utils import get_original_cwd
from tqdm.auto import tqdm
import logging
import torch
import torch.nn.functional as F
from src.model import GAPlanesDVF
from src.registration import warp_volume, compute_jacobian_loss, compute_dvf_tv_loss, DvfScaleStage, DvfLowresSchedule
from src.utils import (
    log_to_json,
    save_checkpoint_slices,
    save_dvf_orthogonal_slices_png,
    save_optimization_frame,
    create_optimization_gif
)

import hydra
from omegaconf import DictConfig, OmegaConf

import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

log = logging.getLogger(__name__)

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config"
)
def main(cfg: DictConfig):
    num_iterations = cfg.training.iterations
    learning_rate = cfg.training.learning_rate
    log_every = cfg.training.log_every

    lambda_jacobian = cfg.loss.lambda_jacobian
    lambda_dvf_tv = cfg.loss.lambda_dvf_tv

    data_root = get_original_cwd()
    moving_path = os.path.join(data_root, cfg.paths.moving)
    target_path = os.path.join(data_root, cfg.paths.target)

    output_dir = os.getcwd()

    frames_dir = os.path.join(output_dir, "frames")
    dvf_dir = os.path.join(output_dir, "dvf_slices")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(dvf_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(moving_path) and os.path.exists(target_path):
        moving = torch.load(moving_path)
        target = torch.load(target_path)
    
        log.info(
            f"Loaded volumes:\n"
            f"  moving = {moving_path}\n"
            f"  target = {target_path}\n"
            f"  shape  = {tuple(moving.shape)}"
        )
    else:
        moving = torch.randn(1, 1, 64, 64, 64)
        target = torch.randn(1, 1, 64, 64, 64)
    
        log.warning(
            "Input volumes not found. Using random tensors:\n"
            f"  moving_path = {moving_path}\n"
            f"  target_path = {target_path}\n"
            f"  shape       = {tuple(moving.shape)}"
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log.info(
        f"Using device: {device}"
        + (
            f" ({torch.cuda.get_device_name(0)})"
            if device.type == "cuda"
            else ""
        )
    )
    moving = moving.to(device)
    target = target.to(device)
    vol_shape = moving.shape[2:]

    model = GAPlanesDVF(
        vol_shape=vol_shape,
        num_shots=1,

        mode=cfg.model.mode,

        Np_list=cfg.model.Np_list,
        Nl_list=cfg.model.Nl_list,
        Nt_list=cfg.model.Nt_list,
        C_list=cfg.model.C_list,

        enable_copies=cfg.model.enable_copies,
        n_copies=cfg.model.n_copies,

        use_trivols=cfg.model.use_trivols,
        use_hypervol=cfg.model.use_hypervol,

        Nvt=cfg.model.Nvt,
        Ntt=cfg.model.Ntt,

        Chv=cfg.model.Chv,
        Nhv=cfg.model.Nhv,
        Nthv=cfg.model.Nthv,

        use_pe=cfg.model.use_pe,
        pe_L_xyz=cfg.model.pe_L_xyz,
        pe_L_t=cfg.model.pe_L_t,

        pe_input_scale=cfg.model.pe_input_scale,

        max_amplitude=cfg.model.max_amplitude,
        output_scale=cfg.model.output_scale,

        decoder_hidden=cfg.model.decoder_hidden,

        init_std_grids=cfg.model.init_std_grids,

        use_chunking=cfg.model.use_chunking,
        spatial_chunk_size=cfg.model.spatial_chunk_size,

        use_gradient_checkpointing=cfg.model.use_gradient_checkpointing
    ).to(device)

    with torch.no_grad():
        u_init = model.materialise_dvf()
        n_params = model.count_parameters()
    
        Np_list = cfg.model.Np_list
        Nl_list = cfg.model.Nl_list
        Nt_list = cfg.model.Nt_list
        C_list = cfg.model.C_list
        use_trivols = cfg.model.use_trivols
        Nvt = cfg.model.Nvt
        Ntt = cfg.model.Ntt
        use_hypervol = cfg.model.use_hypervol
        Chv = cfg.model.Chv
        Nhv = cfg.model.Nhv
        Nthv = cfg.model.Nthv

        log.info(
            "GA-Planes DVF initialized:\n"
            f"  params         = {n_params:,}\n"
            f"  vol_shape      = {vol_shape}\n"
            f"  Np_list        = {Np_list}\n"
            f"  Nl_list        = {Nl_list}\n"
            f"  Nt_list        = {Nt_list}\n"
            f"  C_list         = {C_list}\n"
            f"  use_trivols    = {use_trivols}\n"
            f"  Nvt            = {Nvt}\n"
            f"  Ntt            = {Ntt}\n"
            f"  use_hypervol   = {use_hypervol}\n"
            f"  Chv            = {Chv}\n"
            f"  Nhv            = {Nhv}\n"
            f"  Nthv           = {Nthv}"
        )

        log.info(
            f"Initial DVF u: shape={u_init.shape}, max={u_init.abs().max().item():.4e}"
        )

        del u_init

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = None
    scheduler_cfg = cfg.training.get("lr_scheduler", None)
    if scheduler_cfg and scheduler_cfg.get("enable", False):
        sched_type = scheduler_cfg.get("type", "cosine").lower()
        if sched_type == "cosine":
            eta_min = scheduler_cfg.get("eta_min", 0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_iterations, eta_min=eta_min
            )
            log.info(f"Initialized CosineAnnealingLR scheduler (eta_min={eta_min})")
        elif sched_type == "exponential":
            gamma = scheduler_cfg.get("gamma", 0.99)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )
            log.info(f"Initialized ExponentialLR scheduler (gamma={gamma})")
        else:
            log.warning(f"Unknown scheduler type '{sched_type}'. Running without scheduler.")

    stages = [
        DvfScaleStage(
            until_frac=s.until_frac,
            scale=s.scale
        )
        for s in cfg.schedule.stages
    ]
    
    schedule = DvfLowresSchedule(
        stages=stages,
        max_iter=num_iterations,
        enabled=cfg.schedule.enabled
    )

    run_parameters = {
        "num_iterations": num_iterations,
        "learning_rate": learning_rate,
        "lambda_jacobian": lambda_jacobian,
        "lambda_dvf_tv": lambda_dvf_tv,
        "mode": cfg.model.mode,
        "use_trivols": cfg.model.use_trivols,
        "decoder_hidden": cfg.model.decoder_hidden,
        "vol_shape": list(vol_shape),
        "training": OmegaConf.to_container(cfg.training, resolve=True),
        "loss": OmegaConf.to_container(cfg.loss, resolve=True),
        "schedule": OmegaConf.to_container(cfg.schedule, resolve=True),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
    }

    metrics_log = []
    loss_history = []

    pbar = tqdm(range(num_iterations))
    for it in pbar:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        scale = schedule.get_scale(it)
        if scale < 1.0:
            u_list = model.materialise_dvf_shots_lowres([0], scale=scale)
        else:
            u_list = model.materialise_dvf_shots([0])
        
        dvf = u_list[0]
        warped = warp_volume(moving, dvf)

        loss_mse = F.mse_loss(warped, target)
        loss_jac = compute_jacobian_loss(dvf)
        loss_dvf_tv = compute_dvf_tv_loss(dvf)

        total_loss = loss_mse + lambda_jacobian * loss_jac + lambda_dvf_tv * loss_dvf_tv
        loss_history.append(
            float(total_loss.item())
        )
        total_loss.backward()
        optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']

        if scheduler is not None:
            scheduler.step()

        metrics = {
            "iteration": it,
            "lr": current_lr,
            "mse_loss": float(loss_mse.item()),
            "jacobian_loss": float(loss_jac.item()),
            "dvf_tv_loss": float(loss_dvf_tv.item()),
            "total_loss": float(total_loss.item())
        }
        metrics_log.append(metrics)

        pbar.set_postfix({
            "loss": f"{total_loss.item():.5f}",
            "mse": f"{loss_mse.item():.5f}",
            "lr": f"{current_lr:.2e}"
        })

        if it % log_every == 0 or it == num_iterations - 1:

            model.eval()

            with torch.no_grad():

                u_eval = model.materialise_dvf_shot(0)

                warped_eval = warp_volume(
                    moving,
                    u_eval
                )

                save_checkpoint_slices(
                    os.path.join(
                        checkpoint_dir,
                        f"progress_{it:04d}.png"
                    ),
                    target,
                    moving,
                    warped_eval
                )

                save_dvf_orthogonal_slices_png(
                    u_eval,
                    os.path.join(
                        dvf_dir,
                        f"dvf_{it:04d}.png"
                    )
                )

                save_optimization_frame(
                    os.path.join(
                        frames_dir,
                        f"frame_{it:04d}.png"
                    ),
                    moving,
                    target,
                    warped_eval,
                    loss_history,
                    iteration=it,
                    total_iterations=num_iterations
                )

    log_data = {
        "parameters": run_parameters,
        "metrics": metrics_log
    }
    log_data_path = os.path.join(output_dir, "run_log.json")
    log_to_json(log_data_path, log_data)

    with torch.no_grad():
        final_dvf = model.materialise_dvf_shot(0)

    torch.save(
        final_dvf.cpu(),
        os.path.join(output_dir, "final_dvf.pt")
    )

    log.info(f"Final Loss : {total_loss.item():.6f}")
    log.info(f"Final MSE  : {loss_mse.item():.6f}")
    log.info(f"Final Jac  : {loss_jac.item():.6f}")

    output_gif = os.path.join(output_dir, "optimization.gif")

    create_optimization_gif(
        frames_dir,
        output_gif,
        fps=2
    )
    log.info(f"GIF saved -> {output_gif}")

if __name__ == "__main__":
    main()
