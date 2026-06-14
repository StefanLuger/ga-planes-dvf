
from __future__ import annotations

import argparse
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.registration import warp_volume


def create_random_dvf(
    shape: tuple[int, int, int],
    max_displacement: float = 6.0,
    smooth_factor: int = 8,
) -> torch.Tensor:
    """
    Create a smooth, randomly sampled DVF via low-resolution noise followed
    by trilinear upsampling.

    Args:
        shape:            Spatial dimensions (D, H, W).
        max_displacement: Peak absolute displacement in voxels.
        smooth_factor:    Downsampling factor for the low-resolution noise
                          field. Higher values produce smoother deformations.

    Returns:
        dvf: Tensor of shape (1, 3, D, H, W).
    """
    D, H, W = shape

    dvf_lr = torch.randn(
        1, 3,
        max(D // smooth_factor, 2),
        max(H // smooth_factor, 2),
        max(W // smooth_factor, 2),
    )
    dvf = F.interpolate(dvf_lr, size=(D, H, W), mode="trilinear", align_corners=True)

    peak = dvf.abs().max()
    if peak > 1e-8:
        dvf = dvf / peak
    dvf = dvf * max_displacement

    return dvf


def create_rigid_dvf(
    shape: tuple[int, int, int],
    max_rotation_deg: float = 5.0,
    max_translation_vox: float = 5.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Create a DVF encoding a small random rigid body transform (rotation about
    the volume centre + global translation).

    The rotation is parameterised as intrinsic ZYX Euler angles drawn
    uniformly from [-max_rotation_deg, +max_rotation_deg] for each axis.
    Translations are drawn uniformly from
    [-max_translation_vox, +max_translation_vox] per axis.

    The backward-warp DVF is computed as:

        u(p) = (R - I) p_c  +  t

    where p_c = p - centre is the centred voxel coordinate, R is the combined
    rotation matrix, and t is the translation vector.  This satisfies:

        warped(p) = moving(p + u(p)) = moving(R p_c + centre + t)

    Args:
        shape:                Spatial dimensions (D, H, W).
        max_rotation_deg:     Maximum per-axis rotation magnitude in degrees.
        max_translation_vox:  Maximum per-axis translation magnitude in voxels.
        seed:                 Optional random seed for reproducibility.

    Returns:
        dvf: Tensor of shape (1, 3, D, H, W) with channels (dz, dy, dx).
    """
    rng = random.Random(seed)

    D, H, W = shape

    # Sample random Euler angles (ZYX intrinsic)
    rz = math.radians(rng.uniform(-max_rotation_deg, max_rotation_deg))
    ry = math.radians(rng.uniform(-max_rotation_deg, max_rotation_deg))
    rx = math.radians(rng.uniform(-max_rotation_deg, max_rotation_deg))

    # Sample random translation
    tz = rng.uniform(-max_translation_vox, max_translation_vox)
    ty = rng.uniform(-max_translation_vox, max_translation_vox)
    tx = rng.uniform(-max_translation_vox, max_translation_vox)

    # Elementary rotation matrices acting on (z, y, x) column vectors
    Rz = torch.tensor([
        [math.cos(rz), -math.sin(rz), 0.0],
        [math.sin(rz),  math.cos(rz), 0.0],
        [0.0,           0.0,          1.0],
    ], dtype=torch.float32)

    Ry = torch.tensor([
        [ math.cos(ry), 0.0, math.sin(ry)],
        [ 0.0,          1.0, 0.0         ],
        [-math.sin(ry), 0.0, math.cos(ry)],
    ], dtype=torch.float32)

    Rx = torch.tensor([
        [1.0, 0.0,           0.0          ],
        [0.0, math.cos(rx), -math.sin(rx) ],
        [0.0, math.sin(rx),  math.cos(rx) ],
    ], dtype=torch.float32)

    R = Rz @ Ry @ Rx  # (3, 3) — acts on (z, y, x) column vectors

    t = torch.tensor([tz, ty, tx], dtype=torch.float32)  # (3,)

    # Centred voxel coordinates (z, y, x)
    zz, yy, xx = torch.meshgrid(
        torch.arange(D, dtype=torch.float32) - (D - 1) / 2.0,
        torch.arange(H, dtype=torch.float32) - (H - 1) / 2.0,
        torch.arange(W, dtype=torch.float32) - (W - 1) / 2.0,
        indexing="ij",
    )
    coords = torch.stack([zz, yy, xx], dim=0).reshape(3, -1)  # (3, D*H*W)

    # DVF: u(p) = (R - I) p_c + t
    I = torch.eye(3, dtype=torch.float32)
    disp = (R - I) @ coords + t.unsqueeze(1)              # (3, D*H*W)

    dvf = disp.reshape(3, D, H, W).unsqueeze(0)           # (1, 3, D, H, W)

    print(
        f"Rigid DVF — rotations: ({math.degrees(rz):.2f}°, {math.degrees(ry):.2f}°, "
        f"{math.degrees(rx):.2f}°)  |  translations: ({tz:.2f}, {ty:.2f}, {tx:.2f}) vox"
    )

    return dvf


def create_moving_from_target(
    target_path: str = "data/target_volume.pt",
    output_path: str = "data/moving_volume.pt",
    add_rigid: bool = True,
    max_rotation_deg: float = 5.0,
    max_translation_vox: float = 5.0,
    max_deformable_displacement: float = 8.0,
    noise_std: float = 0.01,
    seed: int | None = 42,
) -> None:
    """
    Generate and save a synthetic moving volume.

    Args:
        target_path:               Path to the reference target tensor.
        output_path:               Destination path for the synthetic moving tensor.
        add_rigid:                 If True, compose a rigid transform with the
                                   non-rigid deformation.
        max_rotation_deg:          Maximum rotation magnitude per axis (degrees).
        max_translation_vox:       Maximum translation magnitude per axis (voxels).
        max_deformable_displacement: Peak displacement for the non-rigid component.
        noise_std:                 Standard deviation of additive Gaussian noise.
        seed:                      Random seed (None = non-deterministic).
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    target = torch.load(target_path)
    D, H, W = target.shape[2:]

    # Non-rigid component
    dvf_deformable = create_random_dvf(
        (D, H, W),
        max_displacement=max_deformable_displacement,
    )

    if add_rigid:
        dvf_rigid = create_rigid_dvf(
            (D, H, W),
            max_rotation_deg=max_rotation_deg,
            max_translation_vox=max_translation_vox,
            seed=seed,
        )
        # Additive composition (valid for small displacements)
        dvf = dvf_deformable + dvf_rigid
        print(
            f"Composed DVF — max deformable: {dvf_deformable.abs().max():.2f} vox  |  "
            f"max rigid: {dvf_rigid.abs().max():.2f} vox  |  "
            f"max total: {dvf.abs().max():.2f} vox"
        )
    else:
        dvf = dvf_deformable
        print(f"Deformable DVF — max displacement: {dvf.abs().max():.2f} vox")

    moving = warp_volume(target, dvf)

    if noise_std > 0.0:
        moving = moving + noise_std * torch.randn_like(moving)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(moving, output_path)
    print(f"Saved synthetic moving volume -> {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a synthetic moving volume from a target volume."
    )
    parser.add_argument(
        "--target",
        default="data/target_volume.pt",
        help="Path to the target volume .pt file.",
    )
    parser.add_argument(
        "--output",
        default="data/moving_volume.pt",
        help="Path to save the synthetic moving volume.",
    )
    parser.add_argument(
        "--no-rigid",
        dest="rigid",
        action="store_false",
        help="Disable the rigid body transform.",
    )
    parser.add_argument(
        "--max_rotation_deg",
        type=float,
        default=8.0,
        help="Maximum per-axis rotation in degrees (only used with --rigid).",
    )
    parser.add_argument(
        "--max_translation_vox",
        type=float,
        default=8.0,
        help="Maximum per-axis translation in voxels (only used with --rigid).",
    )
    parser.add_argument(
        "--max_displacement",
        type=float,
        default=8.0,
        help="Peak displacement for the non-rigid component (voxels).",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.01,
        help="Standard deviation of additive Gaussian noise.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Set to -1 for non-deterministic.",
    )
    args = parser.parse_args()

    create_moving_from_target(
        target_path=args.target,
        output_path=args.output,
        add_rigid=args.rigid,
        max_rotation_deg=args.max_rotation_deg,
        max_translation_vox=args.max_translation_vox,
        max_deformable_displacement=args.max_displacement,
        noise_std=args.noise_std,
        seed=None if args.seed == -1 else args.seed,
    )