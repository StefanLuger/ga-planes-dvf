from __future__ import annotations
from typing import Dict, List, Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

Mode = Literal["nonconvex", "semiconvex", "convex"]

class _NonconvexDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, init_std: float, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, out_dim, bias=True)
        nn.init.normal_(self.fc1.weight, 0.0, 0.1 / max(init_std * (in_dim ** 0.5), 1e-8))
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, 0.0, 0.1 / max(init_std * (hidden ** 0.5), 1e-8))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))

class _SemiconvexDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, hidden, bias=False)
        self.out = nn.Linear(hidden, out_dim, bias=True)
        nn.init.normal_(self.W.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("W_frozen", self.W.weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = (x @ self.W_frozen.T >= 0).float()
        return self.out(self.W(x) * gate)

class _ConvexDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, x: torch.Tensor, x_frozen: torch.Tensor) -> torch.Tensor:
        return self.proj(x * (x_frozen >= 0).float())

class GAPlanesDVF(nn.Module):
    def __init__(
        self,
        vol_shape: Tuple[int, int, int],
        num_shots: int = 1,
        mode: Mode = "nonconvex",
        Np_list: Optional[List[int]] = None,
        Nl_list: Optional[List[int]] = None,
        Nt_list: Optional[List[int]] = None,
        C_list: Optional[List[int]] = None,
        n_copies: Optional[List[int]] = None,
        enable_copies: bool = False,
        use_trivols: bool = True,
        Nvt: int = 8,
        Ntt: int = 16,
        use_hypervol: bool = False,
        Chv: int = 4,
        Nhv: int = 8,
        Nthv: int = 16,
        use_pe: bool = True,
        pe_L_xyz: int = 4,
        pe_L_t: int = 6,
        pe_input_scale: float = 1e-5,
        max_amplitude: float = 40.0,
        output_scale: float = 1.0,
        decoder_hidden: int = 128,
        init_std_grids: float = 0.001,
        use_chunking: bool = False,
        spatial_chunk_size: int = 524288,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if mode not in ("nonconvex", "semiconvex", "convex"):
            raise ValueError(f"Unknown mode: {mode}")
        if Np_list is None: Np_list = [32, 16]
        L = len(Np_list)
        if Nl_list is None: Nl_list = [64, 32]
        if Nt_list is None:
            Nt_list = [1] * L
        if C_list is None: C_list = [8, 8]
        if not (len(Nl_list) == len(Nt_list) == len(C_list) == L):
            raise ValueError("Lists must have equal length.")
        if n_copies is None or not enable_copies:
            n_copies = [1] * L
        self.D, self.H, self.W = vol_shape
        self.T = int(num_shots)
        self.L = L
        self.mode = mode
        self.Np_list = list(Np_list)
        self.Nl_list = list(Nl_list)
        self.Nt_list = list(Nt_list)
        self.C_list = list(C_list)
        self.n_copies = list(n_copies)
        self.use_trivols = bool(use_trivols)
        self.Nvt = int(Nvt)
        self.Ntt = int(Ntt)
        self.use_hypervol = bool(use_hypervol)
        self.Chv = int(Chv)
        self.Nhv = int(Nhv)
        self.Nthv = int(Nthv)
        self.use_pe = bool(use_pe)
        self.pe_L_xyz = int(pe_L_xyz)
        self.pe_L_t = int(pe_L_t)
        self.pe_input_scale = float(pe_input_scale)
        self.max_amplitude = float(max_amplitude)
        self.output_scale = float(output_scale)
        self.init_std_grids = float(init_std_grids)
        self.use_chunking = bool(use_chunking)
        self.spatial_chunk_size = int(spatial_chunk_size)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        if use_pe:
            self._pe_scale = nn.Parameter(torch.tensor(1e-4))

        def _p(*shape: int) -> nn.Parameter:
            if mode == "nonconvex":
                return nn.Parameter(torch.rand(*shape) * 0.0004 + 0.0001)
            return nn.Parameter(float(init_std_grids) * torch.randn(*shape))

        self.lines_mr: nn.ModuleList = nn.ModuleList([
            nn.ModuleList([
                nn.ParameterList([
                    _p(1, C_list[l], Nl_list[l], 1),
                    _p(1, C_list[l], Nl_list[l], 1),
                    _p(1, C_list[l], Nl_list[l], 1),
                    _p(1, C_list[l], Nt_list[l], 1),
                ])
                for _ in range(n_copies[l])
            ])
            for l in range(L)
        ])

        self.planes_mr: nn.ModuleList = nn.ModuleList([
            nn.ModuleList([
                nn.ParameterList([
                    _p(1, C_list[l], Np_list[l], Np_list[l]),
                    _p(1, C_list[l], Np_list[l], Np_list[l]),
                    _p(1, C_list[l], Np_list[l], Np_list[l]),
                    _p(1, C_list[l], Nt_list[l], Np_list[l]),
                    _p(1, C_list[l], Nt_list[l], Np_list[l]),
                    _p(1, C_list[l], Nt_list[l], Np_list[l]),
                ])
                for _ in range(n_copies[l])
            ])
            for l in range(L)
        ])

        if use_trivols:
            self.trivols_mr: nn.ModuleList = nn.ModuleList([
                nn.ParameterList([
                    _p(1, C_list[l], Nvt, Nvt, Nvt),
                    _p(1, C_list[l], Ntt, Nvt, Nvt),
                    _p(1, C_list[l], Ntt, Nvt, Nvt),
                    _p(1, C_list[l], Ntt, Nvt, Nvt),
                ])
                for l in range(L)
            ])

        if use_hypervol:
            self.hypervol = _p(1, Chv, Nthv, Nhv, Nhv, Nhv)
        else:
            self.hypervol = None

        if mode == "nonconvex":
            ppg = 3 if use_trivols else 2
            in_dim = sum(n_copies[l] * ppg * C_list[l] for l in range(L))
        else:
            ppg = 14 if use_trivols else 10
            in_dim = sum(n_copies[l] * ppg * C_list[l] for l in range(L))
            if use_hypervol:
                in_dim += Chv
        if use_pe:
            in_dim += 3 * 2 * pe_L_xyz
            in_dim += 1 * 2 * pe_L_t
        self._in_dim = in_dim

        if mode == "nonconvex":
            self.decoder = _NonconvexDecoder(in_dim, 3, float(init_std_grids), hidden=decoder_hidden)
        elif mode == "semiconvex":
            self.decoder = _SemiconvexDecoder(in_dim, decoder_hidden, 3)
        else:
            self.decoder = _ConvexDecoder(in_dim, 3)

        if mode in ("semiconvex", "convex"):
            self._register_frozen_grids()

        self.register_buffer("_coord_grid_zyx", self._build_coord_grid_zyx())

        if self.T > 1:
            t_vals = torch.tensor([-1.0 + 2.0 * i / (self.T - 1) for i in range(self.T)])
        else:
            t_vals = torch.zeros(self.T)
        self.register_buffer("_t_norms", t_vals)

    def _register_frozen_grids(self) -> None:
        for l in range(self.L):
            for i in range(self.n_copies[l]):
                for a, p in enumerate(self.lines_mr[l][i]):
                    self.register_buffer(f"_frozen_line_l{l}_i{i}_a{a}", p.detach().clone())
                for a, p in enumerate(self.planes_mr[l][i]):
                    self.register_buffer(f"_frozen_plane_l{l}_i{i}_a{a}", p.detach().clone())
        if self.use_trivols:
            for l in range(self.L):
                for a, p in enumerate(self.trivols_mr[l]):
                    self.register_buffer(f"_frozen_trivol_l{l}_a{a}", p.detach().clone())
        if self.use_hypervol and self.hypervol is not None:
            self.register_buffer("_frozen_hypervol", self.hypervol.detach().clone())

    def _get_frozen(self, kind: str, *, l: Optional[int] = None, i: Optional[int] = None, axis: Optional[int] = None) -> torch.Tensor:
        if kind == "line": return getattr(self, f"_frozen_line_l{l}_i{i}_a{axis}")
        if kind == "plane": return getattr(self, f"_frozen_plane_l{l}_i{i}_a{axis}")
        if kind == "trivol": return getattr(self, f"_frozen_trivol_l{l}_a{axis}")
        if kind == "hypervol": return getattr(self, "_frozen_hypervol")
        raise ValueError(kind)

    def _build_coord_grid_zyx(self) -> torch.Tensor:
        D, H, W = self.D, self.H, self.W
        zz, yy, xx = torch.meshgrid(torch.linspace(-1, 1, D), torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        return torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3).unsqueeze(0)

    @staticmethod
    def _sample_line(coord1d: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
        g = torch.stack([coord1d, coord1d], dim=-1).unsqueeze(0)
        f = F.grid_sample(line, g.reshape(1, 1, -1, 2), mode="bilinear", padding_mode="zeros", align_corners=True)
        return f.reshape(1, f.shape[1], -1).permute(0, 2, 1)

    @staticmethod
    def _sample_plane(coords2d: torch.Tensor, plane: torch.Tensor) -> torch.Tensor:
        f = F.grid_sample(plane, coords2d.reshape(1, 1, -1, 2), mode="bilinear", padding_mode="zeros", align_corners=True)
        return f.reshape(1, f.shape[1], -1).permute(0, 2, 1)

    @staticmethod
    def _sample_trivol(coords3d: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
        f = F.grid_sample(vol, coords3d.reshape(1, 1, 1, -1, 3), mode="bilinear", padding_mode="zeros", align_corners=True)
        return f.reshape(1, f.shape[1], -1).permute(0, 2, 1)

    @staticmethod
    def _sample_hypervol_by_t_slices(coords_zyx: torch.Tensor, t: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        Nt = hv.shape[2]
        ti = (t + 1) * (Nt - 1) / 2.0
        t0 = torch.floor(ti).long().clamp(0, Nt - 1)
        t1 = (t0 + 1).clamp(0, Nt - 1)
        w1 = (ti - t0.to(ti.dtype)).clamp(0, 1).view(1, -1, 1)
        w0 = 1.0 - w1
        hv_t = hv.permute(2, 0, 1, 3, 4, 5).contiguous()
        v0 = hv_t.index_select(0, t0).squeeze(1)
        v1 = hv_t.index_select(0, t1).squeeze(1)
        coords_xyz = coords_zyx[..., [2, 1, 0]]
        grid = coords_xyz.permute(1, 0, 2).contiguous().view(-1, 1, 1, 1, 3)
        f0 = F.grid_sample(v0, grid, mode="bilinear", padding_mode="zeros", align_corners=True).view(v0.shape[0], v0.shape[1])
        f1 = F.grid_sample(v1, grid, mode="bilinear", padding_mode="zeros", align_corners=True).view(v1.shape[0], v1.shape[1])
        return ((w0.squeeze(0) * f0) + (w1.squeeze(0) * f1)).unsqueeze(0)

    def _positional_encoding(self, coords: torch.Tensor, L: int = 6) -> torch.Tensor:
        freqs = 2.0 ** torch.arange(L, device=coords.device, dtype=coords.dtype) * torch.pi
        args = coords.unsqueeze(-1) * freqs
        enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return enc.reshape(coords.shape[0], -1)

    def _extract_features(self, coords_zyxt: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = coords_zyxt[0, :, 0]
        y = coords_zyxt[0, :, 1]
        x = coords_zyxt[0, :, 2]
        t = coords_zyxt[0, :, 3]
        coords_xy = torch.stack([x, y], -1).unsqueeze(0)
        coords_xz = torch.stack([x, z], -1).unsqueeze(0)
        coords_yz = torch.stack([y, z], -1).unsqueeze(0)
        coords_xt = torch.stack([x, t], -1).unsqueeze(0)
        coords_yt = torch.stack([y, t], -1).unsqueeze(0)
        coords_zt = torch.stack([z, t], -1).unsqueeze(0)
        live_groups: list[torch.Tensor] = []
        frozen_groups: list[torch.Tensor] = []
        fixed_parts: list[torch.Tensor] = []
        fixed_frozen: list[torch.Tensor] = []
        if self.mode != "nonconvex" and self.use_hypervol and self.hypervol is not None:
            coords_zyx = coords_zyxt[..., :3]
            e_xyzt = self._sample_hypervol_by_t_slices(coords_zyx, t, self.hypervol)
            fixed_parts.append(e_xyzt)
            if self.mode == "convex":
                f_xyzt = self._sample_hypervol_by_t_slices(coords_zyx, t, self._get_frozen("hypervol"))
                fixed_frozen.append(f_xyzt)
        for l in range(self.L):
            for i in range(self.n_copies[l]):
                lines = self.lines_mr[l][i]
                planes = self.planes_mr[l][i]
                e_x = self._sample_line(x, lines[0])
                e_y = self._sample_line(y, lines[1])
                e_z = self._sample_line(z, lines[2])
                e_t = self._sample_line(t, lines[3])
                e_xy = self._sample_plane(coords_xy, planes[0])
                e_xz = self._sample_plane(coords_xz, planes[1])
                e_yz = self._sample_plane(coords_yz, planes[2])
                e_xt = self._sample_plane(coords_xt, planes[3])
                e_yt = self._sample_plane(coords_yt, planes[4])
                e_zt = self._sample_plane(coords_zt, planes[5])
                if self.mode == "nonconvex":
                    LLLL = e_x * e_y * e_z * e_t
                    PP = e_xy * e_zt + e_xz * e_yt + e_xt * e_yz
                    if self.use_trivols:
                        tv = self.trivols_mr[l]
                        e_xyz = self._sample_trivol(torch.stack([x, y, z], -1).unsqueeze(0), tv[0])
                        e_xyt = self._sample_trivol(torch.stack([x, y, t], -1).unsqueeze(0), tv[1])
                        e_xzt = self._sample_trivol(torch.stack([x, z, t], -1).unsqueeze(0), tv[2])
                        e_yzt = self._sample_trivol(torch.stack([y, z, t], -1).unsqueeze(0), tv[3])
                        LVol = e_x * e_yzt + e_y * e_xzt + e_z * e_xyt + e_t * e_xyz
                        live_groups.append(torch.cat([LLLL, LVol, PP], dim=-1))
                    else:
                        live_groups.append(torch.cat([LLLL, PP], dim=-1))
                else:
                    parts = [e_x, e_y, e_z, e_t, e_xy, e_xz, e_yz, e_xt, e_yt, e_zt]
                    if self.use_trivols:
                        tv = self.trivols_mr[l]
                        e_xyz = self._sample_trivol(torch.stack([x, y, z], -1).unsqueeze(0), tv[0])
                        e_xyt = self._sample_trivol(torch.stack([x, y, t], -1).unsqueeze(0), tv[1])
                        e_xzt = self._sample_trivol(torch.stack([x, z, t], -1).unsqueeze(0), tv[2])
                        e_yzt = self._sample_trivol(torch.stack([y, z, t], -1).unsqueeze(0), tv[3])
                        parts += [e_xyz, e_xyt, e_xzt, e_yzt]
                    live_groups.append(torch.cat(parts, dim=-1))
                    if self.mode == "convex":
                        fl = [
                            self._sample_line(x, self._get_frozen("line", l=l, i=i, axis=0)),
                            self._sample_line(y, self._get_frozen("line", l=l, i=i, axis=1)),
                            self._sample_line(z, self._get_frozen("line", l=l, i=i, axis=2)),
                            self._sample_line(t, self._get_frozen("line", l=l, i=i, axis=3)),
                            self._sample_plane(coords_xy, self._get_frozen("plane", l=l, i=i, axis=0)),
                            self._sample_plane(coords_xz, self._get_frozen("plane", l=l, i=i, axis=1)),
                            self._sample_plane(coords_yz, self._get_frozen("plane", l=l, i=i, axis=2)),
                            self._sample_plane(coords_xt, self._get_frozen("plane", l=l, i=i, axis=3)),
                            self._sample_plane(coords_yt, self._get_frozen("plane", l=l, i=i, axis=4)),
                            self._sample_plane(coords_zt, self._get_frozen("plane", l=l, i=i, axis=5)),
                        ]
                        if self.use_trivols:
                            fl += [
                                self._sample_trivol(torch.stack([x, y, z], -1).unsqueeze(0), self._get_frozen("trivol", l=l, axis=0)),
                                self._sample_trivol(torch.stack([x, y, t], -1).unsqueeze(0), self._get_frozen("trivol", l=l, axis=1)),
                                self._sample_trivol(torch.stack([x, z, t], -1).unsqueeze(0), self._get_frozen("trivol", l=l, axis=2)),
                                self._sample_trivol(torch.stack([y, z, t], -1).unsqueeze(0), self._get_frozen("trivol", l=l, axis=3)),
                            ]
                        frozen_groups.append(torch.cat(fl, dim=-1))
        live = torch.cat(live_groups, dim=-1)
        if self.mode != "nonconvex" and fixed_parts:
            live = torch.cat([live, torch.cat(fixed_parts, dim=-1)], dim=-1)
        live = live.squeeze(0)
        if self.use_pe:
            coords_xyz = torch.stack([z, y, x], dim=-1)
            pe_xyz = self._positional_encoding(coords_xyz, L=self.pe_L_xyz)
            pe_t = self._positional_encoding(t.unsqueeze(-1), L=self.pe_L_t)
            scale = self._pe_scale.abs()
            pe_xyz = pe_xyz * scale
            pe_t = pe_t * scale
            live = torch.cat([live, pe_xyz, pe_t], dim=-1)
        if self.mode == "convex":
            frozen = torch.cat(frozen_groups, dim=-1)
            if fixed_frozen:
                frozen = torch.cat([frozen, torch.cat(fixed_frozen, dim=-1)], dim=-1)
            return live, frozen.squeeze(0)
        return live

    def _forward_chunk(self, coord_chunk: torch.Tensor) -> torch.Tensor:
        if self.mode == "convex":
            live, frozen = self._extract_features(coord_chunk)
            out = self.output_scale * self.decoder(live, frozen)
        else:
            out = self.output_scale * self.decoder(self._extract_features(coord_chunk))
        return out.reshape(-1, 3)

    def forward(self, coords_zyxt: torch.Tensor) -> torch.Tensor:
        if coords_zyxt.ndim != 3 or coords_zyxt.shape[-1] != 4:
            raise ValueError()
        B, N, _ = coords_zyxt.shape
        cs = self.spatial_chunk_size
        outs: list[torch.Tensor] = []
        for b in range(B):
            xb = coords_zyxt[b : b + 1]
            if (not self.use_chunking) or N <= cs:
                outs.append(self._forward_chunk(xb))
            else:
                chunks: list[torch.Tensor] = []
                for start in range(0, N, cs):
                    ch = xb[:, start : min(start + cs, N)]
                    if self.use_gradient_checkpointing and self.training:
                        chunks.append(checkpoint(self._forward_chunk, ch, use_reentrant=False))
                    else:
                        chunks.append(self._forward_chunk(ch))
                outs.append(torch.cat(chunks, dim=0))
        out = torch.stack(outs, dim=0)
        return self.max_amplitude * torch.tanh(out / self.max_amplitude)

    def materialise_dvf(self, *, detach: bool = False) -> torch.Tensor:
        if detach:
            with torch.no_grad():
                return self._materialise_dvf_impl().detach()
        return self._materialise_dvf_impl()

    def _materialise_dvf_impl(self) -> torch.Tensor:
        D, H, W, T = self.D, self.H, self.W, self.T
        N = D * H * W
        grid = self._coord_grid_zyx
        dvf_t: list[torch.Tensor] = []
        for shot in range(T):
            t_col = grid.new_full((1, N, 1), float(self._t_norms[shot]))
            coords = torch.cat([grid, t_col], dim=-1)
            disp = self.forward(coords)
            dvf_t.append(disp)
        dvf = torch.stack(dvf_t, dim=3)
        return dvf.reshape(1, D, H, W, 3, T).permute(0, 4, 1, 2, 3, 5)

    def materialise_dvf_shot(self, i: int, *, detach: bool = False) -> torch.Tensor:
        if detach:
            with torch.no_grad():
                return self._materialise_dvf_shot_impl(i).detach()
        return self._materialise_dvf_shot_impl(i)

    def _materialise_dvf_shot_impl(self, i: int) -> torch.Tensor:
        D, H, W = self.D, self.H, self.W
        N = D * H * W
        grid = self._coord_grid_zyx
        t_col = grid.new_full((1, N, 1), float(self._t_norms[i]))
        coords = torch.cat([grid, t_col], dim=-1)
        disp = self.forward(coords)
        return disp.reshape(1, D, H, W, 3).permute(0, 4, 1, 2, 3).contiguous()

    def materialise_dvf_shots(self, shots: list[int], *, detach: bool = False) -> list[torch.Tensor]:
        D, H, W = self.D, self.H, self.W
        N = D * H * W
        grid = self._coord_grid_zyx
        parts = []
        for i in shots:
            t_col = grid.new_full((1, N, 1), float(self._t_norms[i]))
            parts.append(torch.cat([grid, t_col], dim=-1))
        all_coords = torch.cat(parts, dim=1)
        if detach:
            with torch.no_grad():
                disp = self.forward(all_coords).detach()
        else:
            disp = self.forward(all_coords)
        disp_flat = disp.squeeze(0)
        result: list[torch.Tensor] = []
        for k in range(len(shots)):
            u_k = disp_flat[k * N : (k + 1) * N].reshape(D, H, W, 3).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
            result.append(u_k)
        return result

    def materialise_dvf_shots_lowres(self, shots: list[int], scale: float = 0.25, *, detach: bool = False) -> list[torch.Tensor]:
        D, H, W = self.D, self.H, self.W
        D_lr = max(int(D * scale), 4)
        H_lr = max(int(H * scale), 4)
        W_lr = max(int(W * scale), 4)
        N_lr = D_lr * H_lr * W_lr
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, D_lr, device=self._coord_grid_zyx.device),
            torch.linspace(-1, 1, H_lr, device=self._coord_grid_zyx.device),
            torch.linspace(-1, 1, W_lr, device=self._coord_grid_zyx.device),
            indexing="ij",
        )
        grid_lr = torch.stack([zz, yy, xx], dim=-1).reshape(1, N_lr, 3)
        parts = []
        for i in shots:
            t_col = grid_lr.new_full((1, N_lr, 1), float(self._t_norms[i]))
            parts.append(torch.cat([grid_lr, t_col], dim=-1))
        all_coords = torch.cat(parts, dim=1)
        if detach:
            with torch.no_grad():
                disp = self.forward(all_coords).detach()
        else:
            disp = self.forward(all_coords)
        disp_flat = disp.squeeze(0)
        result: list[torch.Tensor] = []
        for k in range(len(shots)):
            u_lr = disp_flat[k * N_lr : (k + 1) * N_lr].reshape(D_lr, H_lr, W_lr, 3).permute(3, 0, 1, 2).unsqueeze(0)
            u_full = F.interpolate(u_lr, size=(D, H, W), mode="trilinear", align_corners=True)
            result.append(u_full)
        return result

    def spatial_tv_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        eps = 1e-12
        def tv2d(p: torch.Tensor) -> torch.Tensor:
            return (torch.sqrt(((p[:, :, 1:] - p[:, :, :-1]) ** 2).sum() + eps) + torch.sqrt(((p[:, :, :, 1:] - p[:, :, :, :-1]) ** 2).sum() + eps))
        def tv1d(p: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(((p[:, :, 1:] - p[:, :, :-1]) ** 2).sum() + eps)
        def tv3d(p: torch.Tensor) -> torch.Tensor:
            return (torch.sqrt(((p[:, :, 1:] - p[:, :, :-1]) ** 2).sum() + eps) + torch.sqrt(((p[:, :, :, 1:] - p[:, :, :, :-1]) ** 2).sum() + eps) + torch.sqrt(((p[:, :, :, :, 1:] - p[:, :, :, :, :-1]) ** 2).sum() + eps))
        for l in range(self.L):
            for i in range(self.n_copies[l]):
                for p in self.lines_mr[l][i]: loss = loss + tv1d(p)
                for p in self.planes_mr[l][i]: loss = loss + tv2d(p)
        if self.use_trivols:
            for l in range(self.L):
                for p in self.trivols_mr[l]: loss = loss + tv3d(p)
        return loss

    def temporal_smoothness_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        def acc2(p: torch.Tensor, d: int) -> torch.Tensor:
            sl = [slice(None)] * p.ndim
            def s(a, b):
                r = list(sl); r[d] = slice(a, b); return tuple(r)
            return ((p[s(2, None)] - 2 * p[s(1, -1)] + p[s(None, -2)]) ** 2).sum()
        for l in range(self.L):
            for i in range(self.n_copies[l]):
                et = self.lines_mr[l][i][3]
                if et.shape[2] >= 3: loss = loss + acc2(et, 2)
                for k in (3, 4, 5):
                    p = self.planes_mr[l][i][k]
                    if p.shape[2] >= 3: loss = loss + acc2(p, 2)
        if self.use_trivols:
            for l in range(self.L):
                for k in (1, 2, 3):
                    tv = self.trivols_mr[l][k]
                    if tv.shape[2] >= 3: loss = loss + acc2(tv, 2)
        return loss


    @torch.no_grad()
    def dvf_stats(self) -> dict:
        dvf      = self.materialise_dvf()               # (1, 3, D, H, W, T)
        mag      = dvf.pow(2).sum(dim=1).sqrt()          # (1, D, H, W, T)
        per_shot = mag.amax(dim=(0, 1, 2, 3))            # (T,)
        return {
            "dvf/max_shot_mean": float(per_shot.mean()),
            "dvf/max_shot_max":  float(per_shot.max()),
            "in_dim":            int(self._in_dim),
            "mode":              self.mode,
            "use_trivols":       self.use_trivols,
            "use_hypervol":      self.use_hypervol,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)