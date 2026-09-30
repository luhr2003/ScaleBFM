"""Terrain layouts for ScaleTrack: import recorded terrain meshes next to the flat ground plane and query their height.

A *layout* is one terrain mesh exported by the terrain recorder (`<root>/layouts/layout_<seed>/terrain.npz` with
`vertices`/`faces`, and `heightmap.npz` with a regular height grid `height[nx, ny]`, `x0`, `y0`, `res`). Terrain clips
are stored in the layout's own frame. In the simulator every layout is translated by a fixed world offset

    offset_k = (origin_x, origin_y + k * pitch_y, z_lift)

(`z_lift` keeps the deepest pit above the infinite ground plane that the flat clips use), and an env that replays a clip of
layout k adds `offset_k` to all reference positions instead of its grid origin.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import trimesh

from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass


def layout_offsets(origin_xy: tuple[float, float], pitch_y: float, z_lift: float, num: int) -> np.ndarray:
    return np.array([[origin_xy[0], origin_xy[1] + k * pitch_y, z_lift] for k in range(num)], dtype=np.float64)


class LayoutTerrainImporter(TerrainImporter):
    """Ground plane (env grid as usual) plus the recorded terrain layouts imported as static colliders."""

    def __init__(self, cfg: "LayoutTerrainImporterCfg"):
        super().__init__(cfg)
        self.layout_offsets = layout_offsets(cfg.layout_origin_xy, cfg.layout_pitch_y, cfg.layout_z_lift, len(cfg.layout_seeds))
        for k, seed in enumerate(cfg.layout_seeds):
            path = os.path.join(cfg.layout_root, "layouts", f"layout_{seed}", "terrain.npz")
            z = np.load(path)
            vertices = z["vertices"].astype(np.float64) + self.layout_offsets[k]
            mesh = trimesh.Trimesh(vertices=vertices, faces=z["faces"].astype(np.int64), process=False)
            self.import_mesh(f"layout_{seed}", mesh)
            print(f"[terrain] layout {seed}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, "
                  f"offset {self.layout_offsets[k].tolist()}", flush=True)


@configclass
class LayoutTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = LayoutTerrainImporter
    layout_root: str = ""
    """Directory that contains `layouts/layout_<seed>/`."""
    layout_seeds: list[int] = []
    layout_origin_xy: tuple[float, float] = (-320.0, 0.0)
    layout_pitch_y: float = 260.0
    layout_z_lift: float = 6.0


class LayoutHeightSampler:
    """Bilinear ground-height queries (world frame) over the exported layout height grids."""

    def __init__(self, layout_root: str, layout_seeds: list[int], offsets: np.ndarray, device):
        self.device = device
        grids, meta = [], []
        for seed in layout_seeds:
            h = np.load(os.path.join(layout_root, "layouts", f"layout_{seed}", "heightmap.npz"))
            grids.append(torch.from_numpy(h["height"].astype(np.float32)))
            meta.append((float(h["x0"]), float(h["y0"]), float(h["res"])))
        assert len({m[2] for m in meta}) <= 1, "all layouts must share the same heightmap resolution"
        self.res = meta[0][2] if meta else 1.0
        self.grids = torch.stack(grids).to(device) if grids else torch.zeros(0, 1, 1, device=device)  # (K, nx, ny)
        self.x0 = torch.tensor([m[0] for m in meta], device=device, dtype=torch.float32)
        self.y0 = torch.tensor([m[1] for m in meta], device=device, dtype=torch.float32)
        self.offsets = torch.as_tensor(offsets, dtype=torch.float32, device=device)

    def height(self, xy_w: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        """World-frame terrain height under `xy_w` (N,2) for envs on layout index `layout` (N,), 0 on the flat plane."""
        out = torch.zeros(xy_w.shape[0], device=xy_w.device)
        on = layout >= 0
        if not bool(on.any()) or self.grids.shape[0] == 0:
            return out
        k = layout[on]
        loc = xy_w[on] - self.offsets[k, :2]
        fx = (loc[:, 0] - self.x0[k]) / self.res
        fy = (loc[:, 1] - self.y0[k]) / self.res
        nx, ny = self.grids.shape[1], self.grids.shape[2]
        fx = fx.clamp(0, nx - 1.001)
        fy = fy.clamp(0, ny - 1.001)
        ix, iy = fx.floor().long(), fy.floor().long()
        tx, ty = fx - ix, fy - iy
        g = self.grids
        h00, h10 = g[k, ix, iy], g[k, ix + 1, iy]
        h01, h11 = g[k, ix, iy + 1], g[k, ix + 1, iy + 1]
        h = (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty
        out[on] = h + self.offsets[k, 2]
        return out
