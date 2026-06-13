import os
import torch
import torch.nn.functional as F
from src.model import GAPlanesDVF
from src.registration import warp_volume, compute_jacobian_loss, compute_dvf_tv_loss, DvfScaleStage, DvfLowresSchedule
from src.utils import log_to_json, save_checkpoint_slices

def main():
    num_iterations = 200
    learning_rate = 1e-3
    lambda_jacobian = 1e3
    lambda_grid_tv = 1e-1
    lambda_dvf_tv = 1e-1
    mode = "nonconvex"
    use_trivols = True
    decoder_hidden = 128
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    moving_path = "data/moving_volume.pt"
    target_path = "data/target_volume.pt"

    if os.path.exists(moving_path) and os.path.exists(target_path):
        moving = torch.load(moving_path)
        target = torch.load(target_path)
    else:
        moving = torch.randn(1, 1, 64, 64, 64)
        target = torch.randn(1, 1, 64, 64, 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moving = moving.to(device)
    target = target.to(device)
    vol_shape = moving.shape[2:]

    model = GAPlanesDVF(
        vol_shape=vol_shape,
        num_shots=1,
        mode=mode,
        use_trivols=use_trivols,
        decoder_hidden=decoder_hidden
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    stages = [
        DvfScaleStage(until_frac=0.4, scale=0.25),
        DvfScaleStage(until_frac=0.7, scale=0.5),
        DvfScaleStage(until_frac=1.0, scale=1.0)
    ]
    schedule = DvfLowresSchedule(stages=stages, max_iter=num_iterations, enabled=True)

    run_parameters = {
        "num_iterations": num_iterations,
        "learning_rate": learning_rate,
        "lambda_jacobian": lambda_jacobian,
        "lambda_grid_tv": lambda_grid_tv,
        "lambda_dvf_tv": lambda_dvf_tv,
        "mode": mode,
        "use_trivols": use_trivols,
        "decoder_hidden": decoder_hidden,
        "vol_shape": list(vol_shape)
    }

    metrics_log = []

    for it in range(num_iterations):
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
        loss_grid_tv = model.spatial_tv_loss()
        loss_dvf_tv = compute_dvf_tv_loss(dvf)

        total_loss = loss_mse + lambda_jacobian * loss_jac + lambda_grid_tv * loss_grid_tv + lambda_dvf_tv * loss_dvf_tv
        total_loss.backward()
        optimizer.step()

        metrics = {
            "iteration": it,
            "mse_loss": float(loss_mse.item()),
            "jacobian_loss": float(loss_jac.item()),
            "grid_tv_loss": float(loss_grid_tv.item()),
            "dvf_tv_loss": float(loss_dvf_tv.item()),
            "total_loss": float(total_loss.item())
        }
        metrics_log.append(metrics)

        if it % 10 == 0 or it == num_iterations - 1:
            model.eval()
            with torch.no_grad():
                u_eval = model.materialise_dvf_shot(0)
                warped_eval = warp_volume(moving, u_eval)
                
                slice_path = os.path.join(output_dir, f"progress_iter_{it:04d}.png")
                save_checkpoint_slices(slice_path, target, moving, warped_eval)
                
                dvf_path = os.path.join(output_dir, f"dvf_iter_{it:04d}.pt")
                torch.save(u_eval.cpu(), dvf_path)

    log_data = {
        "parameters": run_parameters,
        "metrics": metrics_log
    }
    log_data_path = os.path.join(output_dir, "run_log.json")
    log_to_json(log_data_path, log_data)

if __name__ == "__main__":
    main()