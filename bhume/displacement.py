"""Global and local adaptive displacement field estimation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from bhume.potential import FusedPotential


@dataclass
class DisplacementField:
    """Continuous regularized 2D displacement vector field (u(x, y), v(x, y))."""
    tile_centers: np.ndarray    # (K, 2)
    tile_dxs: np.ndarray        # (K,)
    tile_dys: np.ndarray        # (K,)
    tile_weights: np.ndarray    # (K,)
    global_dx: float = 0.0
    global_dy: float = 0.0
    sigma_m: float = 500.0

    def query(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized query of displacement (u, v) at coordinates (xs, ys) in EPSG:3857."""
        if len(self.tile_centers) == 0:
            return np.full_like(xs, self.global_dx), np.full_like(ys, self.global_dy)

        # Distance matrix (N, K)
        dx_mat = xs[:, None] - self.tile_centers[:, 0][None, :]
        dy_mat = ys[:, None] - self.tile_centers[:, 1][None, :]
        dists_sq = dx_mat**2 + dy_mat**2

        # Kernel weights
        k_weights = np.exp(-dists_sq / (2.0 * self.sigma_m**2)) * self.tile_weights[None, :]
        sum_w = np.sum(k_weights, axis=1)

        # Strong prior toward coherent global translation
        prior_w = 1.0
        u = (np.sum(k_weights * self.tile_dxs[None, :], axis=1) + prior_w * self.global_dx) / (sum_w + prior_w)
        v = (np.sum(k_weights * self.tile_dys[None, :], axis=1) + prior_w * self.global_dy) / (sum_w + prior_w)
        return u.astype(np.float32), v.astype(np.float32)


def sample_perimeter_coords(geom, step_m: float = 2.0) -> np.ndarray:
    """Extract 2D coordinates sampled along exterior boundary of a geometry."""
    pts = []
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for poly in polys:
        ext = poly.exterior
        length = ext.length
        if length <= 0:
            continue
        num_pts = max(4, int(np.ceil(length / step_m)))
        dists = np.linspace(0, length, num_pts, endpoint=False)
        for d in dists:
            p = ext.interpolate(d)
            pts.append((p.x, p.y))
    return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)


def estimate_displacement_field(
    plots_3857: gpd.GeoDataFrame,
    potential: FusedPotential,
    tile_size_m: float = 500.0,
    stride_m: float = 250.0,
) -> DisplacementField:
    """Estimate a robust global translation and regularized low-rank residual displacement field.

    1. Identifies confident anchor plots across the village to establish a single coherent
       primary global translation (u0, v0).
    2. Fits smoothly varying residual adjustments around (u0, v0) with strong regularization
       to prevent topological tearing and aliasing.
    """
    centroids = plots_3857.geometry.centroid
    cx_all = centroids.x.to_numpy()
    cy_all = centroids.y.to_numpy()
    bounds = plots_3857.total_bounds

    # Pre-extract sample points per plot
    plot_pts = {
        pn: sample_perimeter_coords(geom, step_m=2.0)
        for pn, geom in zip(plots_3857.index, plots_3857.geometry)
    }

    # Step 1: Robust Whole-Village Global Translation Estimation
    n_sample = min(600, len(plots_3857))
    sample_indices = plots_3857.sample(n_sample, random_state=42).index

    cand_global = [(dx, dy) for dx in range(-20, 21, 2) for dy in range(-20, 21, 2)]
    cg_arr = np.array(cand_global, dtype=np.float32)

    anchor_dxs = []
    anchor_dys = []

    for pn in sample_indices:
        pts = plot_pts[pn]
        if len(pts) < 10:
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        sc0 = float(np.mean(potential.sample_coords(xs, ys)))

        all_x = (xs[:, None] + cg_arr[:, 0][None, :]).ravel()
        all_y = (ys[:, None] + cg_arr[:, 1][None, :]).ravel()
        scs = potential.sample_coords(all_x, all_y).reshape(len(xs), len(cand_global)).mean(axis=0)

        # Light regularization against extreme jumps during anchor search
        penalized = scs - 0.0005 * (cg_arr[:, 0]**2 + cg_arr[:, 1]**2)
        b_idx = int(np.argmax(penalized))
        b_sc = float(scs[b_idx])

        # Confident anchor criterion: high peak score and notable gain
        if b_sc >= 0.35 and (b_sc - sc0) >= 0.10:
            anchor_dxs.append(cand_global[b_idx][0])
            anchor_dys.append(cand_global[b_idx][1])

    if len(anchor_dxs) >= 10:
        global_dx = float(np.median(anchor_dxs))
        global_dy = float(np.median(anchor_dys))
    else:
        global_dx = 0.0
        global_dy = 0.0

    # Step 2: Smooth low-rank residual tile corrections around (global_dx, global_dy)
    xs_tiles = np.arange(bounds[0], bounds[2] + tile_size_m, stride_m)
    ys_tiles = np.arange(bounds[1], bounds[3] + tile_size_m, stride_m)

    delta_cands = [(dx, dy) for dx in np.arange(-3.0, 3.5, 1.0) for dy in np.arange(-3.0, 3.5, 1.0)]
    d_arr = np.array(delta_cands, dtype=np.float32)

    tile_centers = []
    tile_dxs = []
    tile_dys = []
    tile_weights = []

    for tx in xs_tiles:
        for ty in ys_tiles:
            in_tile = (
                (cx_all >= tx - tile_size_m / 2)
                & (cx_all < tx + tile_size_m / 2)
                & (cy_all >= ty - tile_size_m / 2)
                & (cy_all < ty + tile_size_m / 2)
            )
            sub_indices = plots_3857.index[in_tile]
            if len(sub_indices) < 8:
                continue

            t_pts_list = [plot_pts[pn] for pn in sub_indices if len(plot_pts[pn]) > 0]
            if not t_pts_list:
                continue
            t_pts = np.vstack(t_pts_list)
            if len(t_pts) < 150:
                continue
            if len(t_pts) > 2000:
                t_pts = t_pts[np.random.RandomState(42).choice(len(t_pts), 2000, replace=False)]

            xs, ys = t_pts[:, 0], t_pts[:, 1]
            base_sc = float(np.mean(potential.sample_coords(xs + global_dx, ys + global_dy)))

            all_x = (xs[:, None] + (global_dx + d_arr[:, 0])[None, :]).ravel()
            all_y = (ys[:, None] + (global_dy + d_arr[:, 1])[None, :]).ravel()
            scs = potential.sample_coords(all_x, all_y).reshape(len(xs), len(delta_cands)).mean(axis=0)

            # Strong quadratic penalty against local deviations from the global translation
            penalized = scs - 0.003 * (d_arr[:, 0]**2 + d_arr[:, 1]**2)
            b_idx = int(np.argmax(penalized))
            b_sc = float(scs[b_idx])
            b_dx = float(d_arr[b_idx, 0])
            b_dy = float(d_arr[b_idx, 1])

            gain = b_sc - base_sc
            dist_dev = np.hypot(b_dx, b_dy)

            # Restraint threshold for local deviations
            if gain >= 0.03 and b_sc >= 0.25 and dist_dev >= 0.8:
                w = float(np.clip(gain * 5.0, 0.4, 2.0))
                tile_dxs.append(global_dx + b_dx)
                tile_dys.append(global_dy + b_dy)
            else:
                w = 0.5
                tile_dxs.append(global_dx)
                tile_dys.append(global_dy)

            tile_centers.append((tx, ty))
            tile_weights.append(w)

    return DisplacementField(
        tile_centers=np.array(tile_centers, dtype=np.float32) if tile_centers else np.empty((0, 2), dtype=np.float32),
        tile_dxs=np.array(tile_dxs, dtype=np.float32) if tile_dxs else np.empty(0, dtype=np.float32),
        tile_dys=np.array(tile_dys, dtype=np.float32) if tile_dys else np.empty(0, dtype=np.float32),
        tile_weights=np.array(tile_weights, dtype=np.float32) if tile_weights else np.empty(0, dtype=np.float32),
        global_dx=global_dx,
        global_dy=global_dy,
        sigma_m=500.0,
    )
