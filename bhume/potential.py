"""Continuous boundary potential field computation from boundaries.tif and imagery.tif."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel


@dataclass
class FusedPotential:
    """Continuous 2D boundary attraction field in EPSG:3857."""
    grid: np.ndarray        # (H, W) float32 in [0, 1]
    transform: rasterio.Affine
    inv_transform: rasterio.Affine
    crs: str
    res: float

    def sample_coords(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Sample potential at spatial coordinates (xs, ys) in EPSG:3857."""
        # Map world (x, y) to fractional pixel (col, row)
        cols = (xs - self.transform.c) / self.transform.a
        rows = (ys - self.transform.f) / self.transform.e

        # Clamp to valid image bounds
        H, W = self.grid.shape
        cols = np.clip(cols, 0, W - 1.001)
        rows = np.clip(rows, 0, H - 1.001)

        # Bilinear interpolation
        c0 = np.floor(cols).astype(int)
        r0 = np.floor(rows).astype(int)
        c1 = np.clip(c0 + 1, 0, W - 1)
        r1 = np.clip(r0 + 1, 0, H - 1)

        dc = cols - c0
        dr = rows - r0

        v00 = self.grid[r0, c0]
        v01 = self.grid[r0, c1]
        v10 = self.grid[r1, c0]
        v11 = self.grid[r1, c1]

        val = (1 - dr) * ((1 - dc) * v00 + dc * v01) + dr * ((1 - dc) * v10 + dc * v11)
        return val

    def perimeter_score(self, pts_xy: np.ndarray) -> float:
        """Mean potential along a sequence of perimeter coordinates (N, 2)."""
        if len(pts_xy) == 0:
            return 0.0
        vals = self.sample_coords(pts_xy[:, 0], pts_xy[:, 1])
        return float(np.mean(vals))


def build_potential_field(
    boundaries_path: Path | str | None,
    imagery_path: Path | str | None,
    sigma_bnd_m: float = 2.5,
    sigma_img_m: float = 2.0,
) -> FusedPotential:
    """Build a continuous, normalized boundary potential grid from rasters.

    Combines:
    1. Distance-transformed boundaries.tif (if present)
    2. Edge gradients from imagery.tif (if present)
    """
    p_bnd = None
    ref_transform = None
    ref_crs = None
    res = 1.2

    # 1. Process boundaries.tif if present
    if boundaries_path and Path(boundaries_path).exists():
        with rasterio.open(boundaries_path) as src:
            bnd_data = src.read(1)
            ref_transform = src.transform
            ref_crs = str(src.crs)
            res = float(abs(src.transform.a))

            # Binary mask: non-zero pixels are edges
            edge_mask = bnd_data > 0
            if np.any(edge_mask):
                # Distance transform in pixels
                dt_px = distance_transform_edt(~edge_mask)
                dt_m = dt_px * res
                eff_sigma = max(sigma_bnd_m, 1.25 * res)
                # Gaussian potential: 1.0 on edge, smoothly decays to 0
                grid = np.exp(-(dt_m**2) / (2.0 * eff_sigma**2)).astype(np.float32)
            else:
                grid = np.zeros(bnd_data.shape, dtype=np.float32)

    # 2. Fallback to imagery.tif only if boundaries.tif is not available
    elif imagery_path and Path(imagery_path).exists():
        with rasterio.open(imagery_path) as isrc:
            ref_transform = isrc.transform
            ref_crs = str(isrc.crs)
            res = float(abs(isrc.transform.a))
            target_shape = (isrc.height, isrc.width)

            rgb = isrc.read(
                [1, 2, 3],
                out_shape=(3, target_shape[0], target_shape[1]),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)

            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            exg = 2.0 * rgb[1] - rgb[0] - rgb[2]

            gx_l = sobel(lum, axis=1)
            gy_l = sobel(lum, axis=0)
            gx_v = sobel(exg, axis=1)
            gy_v = sobel(exg, axis=0)

            grad_mag = np.sqrt(gx_l**2 + gy_l**2 + gx_v**2 + gy_v**2)
            grad_smooth = gaussian_filter(grad_mag, sigma=sigma_img_m / res)
            p98 = np.percentile(grad_smooth, 98)
            if p98 > 0:
                grid = np.clip(grad_smooth / p98, 0.0, 1.0).astype(np.float32)
            else:
                grid = np.zeros(target_shape, dtype=np.float32)
    else:
        raise ValueError("Neither boundaries.tif nor imagery.tif could be loaded")

    # Normalize grid to [0, 1]
    g_min, g_max = float(grid.min()), float(grid.max())
    if g_max > g_min:
        grid = (grid - g_min) / (g_max - g_min)

    return FusedPotential(
        grid=grid,
        transform=ref_transform,
        inv_transform=~ref_transform,
        crs=ref_crs or "EPSG:3857",
        res=res,
    )
