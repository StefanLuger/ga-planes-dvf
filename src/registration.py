from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from src.model import GAPlanesDVF

@dataclass
class DvfScaleStage:
    until_frac: float
    scale: float

class DvfLowresSchedule:
    def __init__(self, stages: List[DvfScaleStage], max_iter: int, enabled: bool = True, fallback_scale: float = 1.0):
        self.stages = sorted(stages, key=lambda s: s.until_frac)
        self.max_iter = max_iter
        self.enabled = enabled
        self.fallback = fallback_scale

    def get_scale(self, current_iter: int) -> float:
        if not self.enabled:
            return self.fallback
        frac = current_iter / max(self.max_iter - 1, 1)
        for stage in self.stages:
            if frac <= stage.until_frac:
                return stage.scale
        return self.stages[-1].scale

def warp_volume(moving: torch.Tensor, dvf: torch.Tensor) -> torch.Tensor:
    D, H, W = moving.shape[2], moving.shape[3], moving.shape[4]
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, D, device=moving.device),
        torch.linspace(-1, 1, H, device=moving.device),
        torch.linspace(-1, 1, W, device=moving.device),
        indexing="ij"
    )
    grid_identity = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)
    u_z = dvf[:, 0] * (2.0 / max(D - 1, 1))
    u_y = dvf[:, 1] * (2.0 / max(H - 1, 1))
    u_x = dvf[:, 2] * (2.0 / max(W - 1, 1))
    dvf_normalized = torch.stack([u_x, u_y, u_z], dim=-1)
    grid = grid_identity + dvf_normalized
    return F.grid_sample(moving, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

def compute_jacobian_loss(u: torch.Tensor, eps: float = 1e-6, fold_weight: float = 10.0) -> torch.Tensor:
    duz_dz = u[:, 0, 1:, :-1, :-1] - u[:, 0, :-1, :-1, :-1]
    duz_dy = u[:, 0, :-1, 1:, :-1] - u[:, 0, :-1, :-1, :-1]
    duz_dx = u[:, 0, :-1, :-1, 1:] - u[:, 0, :-1, :-1, :-1]
    duy_dz = u[:, 1, 1:, :-1, :-1] - u[:, 1, :-1, :-1, :-1]
    duy_dy = u[:, 1, :-1, 1:, :-1] - u[:, 1, :-1, :-1, :-1]
    duy_dx = u[:, 1, :-1, :-1, 1:] - u[:, 1, :-1, :-1, :-1]
    dux_dz = u[:, 2, 1:, :-1, :-1] - u[:, 2, :-1, :-1, :-1]
    dux_dy = u[:, 2, :-1, 1:, :-1] - u[:, 2, :-1, :-1, :-1]
    dux_dx = u[:, 2, :-1, :-1, 1:] - u[:, 2, :-1, :-1, :-1]
    j11 = 1.0 + duz_dz
    j12 = duz_dy
    j13 = duz_dx
    j21 = duy_dz
    j22 = 1.0 + duy_dy
    j23 = duy_dx
    j31 = dux_dz
    j32 = dux_dy
    j33 = 1.0 + dux_dx
    det_J = (
        j11 * (j22 * j33 - j23 * j32) -
        j12 * (j21 * j33 - j23 * j31) +
        j13 * (j21 * j32 - j22 * j31)
    )
    det_J_safe = torch.clamp(det_J, min=eps)
    smooth_loss = torch.mean(torch.log(det_J_safe) ** 2)
    fold_loss = torch.mean(torch.clamp(-det_J, min=0.0))
    return smooth_loss + fold_weight * fold_loss

def compute_dvf_tv_loss(u: torch.Tensor) -> torch.Tensor:
    diff_z = torch.pow(u[:, :, 1:, :, :] - u[:, :, :-1, :, :], 2).sum()
    diff_y = torch.pow(u[:, :, :, 1:, :] - u[:, :, :, :-1, :], 2).sum()
    diff_x = torch.pow(u[:, :, :, :, 1:] - u[:, :, :, :, :-1], 2).sum()
    return (diff_z + diff_y + diff_x) / u.numel()