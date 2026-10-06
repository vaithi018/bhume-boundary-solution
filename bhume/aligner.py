"""Plot-level boundary optimization, restraint gating, confidence scoring, and flagging."""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.affinity import translate

from bhume.displacement import DisplacementField, sample_perimeter_coords
from bhume.io import Village
from bhume.potential import FusedPotential


def align_plots(
    village: Village,
    potential: FusedPotential,
    field: DisplacementField,
    search_radius_m: float = 2.5,
    search_step_m: float = 0.5,
    lambda_restraint: float = 0.0015,
    min_score: float = 0.25,
    min_gain: float = 0.05,
    min_shift_m: float = 0.6,
    flag_threshold: float = 0.45,
) -> gpd.GeoDataFrame:
    """Run plot-level boundary alignment with quadratic restraint gating and calibrated confidence.

    - Protects already-aligned plots with quadratic movement penalties and restraint gating.
    - Preserves exact input geometry for every flagged or low-confidence plot.
    - Emits corrected status only when boundary evidence provides confident improvement.
    """
    plots_3857 = village.plots.to_crs("EPSG:3857")
    centroids = plots_3857.geometry.centroid
    cx_all = centroids.x.to_numpy()
    cy_all = centroids.y.to_numpy()

    # Query regularized displacement field
    u_field, v_field = field.query(cx_all, cy_all)

    # Offset grid for fine search around local field prediction
    offsets = np.arange(-search_radius_m, search_radius_m + 0.01, search_step_m, dtype=np.float32)
    ox, oy = np.meshgrid(offsets, offsets)
    ox_flat = ox.ravel()
    oy_flat = oy.ravel()

    results = []
    corrected_indices = []
    shifted_geoms_3857 = []

    for idx, (pn, geom_3857, geom_4326) in enumerate(
        zip(village.plots.index, plots_3857.geometry, village.plots.geometry)
    ):
        pts = sample_perimeter_coords(geom_3857, step_m=2.0)
        if len(pts) == 0:
            results.append({
                "plot_number": str(pn),
                "status": "flagged",
                "confidence": None,
                "method_note": "empty geometry; official retained",
                "geometry": geom_4326,
            })
            continue

        xs = pts[:, 0]
        ys = pts[:, 1]
        N = len(pts)

        # Baseline score of official geometry (dx=0, dy=0)
        sc_official = float(np.mean(potential.sample_coords(xs, ys)))

        # Candidate shifts: unshifted (0, 0) plus fine grid centered at (u0, v0)
        u0 = float(u_field[idx])
        v0 = float(v_field[idx])

        cand_x = np.concatenate([[0.0], u0 + ox_flat])
        cand_y = np.concatenate([[0.0], v0 + oy_flat])
        num_cand = len(cand_x)

        all_query_x = (xs[:, None] + cand_x[None, :]).ravel()
        all_query_y = (ys[:, None] + cand_y[None, :]).ravel()
        sampled = potential.sample_coords(all_query_x, all_query_y).reshape(N, num_cand)
        cand_scores = sampled.mean(axis=0)

        # Quadratic restraint penalty: penalize distance moved from official placement
        penalized = cand_scores - lambda_restraint * (cand_x**2 + cand_y**2)
        best_idx = int(np.argmax(penalized))

        best_sc = float(cand_scores[best_idx])
        best_dx = float(cand_x[best_idx])
        best_dy = float(cand_y[best_idx])

        gain = best_sc - sc_official
        shift_dist = float(np.hypot(best_dx, best_dy))

        # Restraint gating:
        # Require substantial boundary score, genuine gain over official, and meaningful movement
        is_restrained = (
            (shift_dist < min_shift_m)
            or (best_sc < min_score)
            or (gain < min_gain)
            or (sc_official >= 0.35 and gain < 0.08)
        )

        if is_restrained:
            results.append({
                "plot_number": str(pn),
                "status": "flagged",
                "confidence": None,
                "method_note": f"restrained (sc0={sc_official:.2f}, peak={best_sc:.2f}); official retained",
                "geometry": geom_4326,  # EXACT official geometry preserved
            })
        else:
            # Calibrated Confidence Formulation
            c_edge = float(np.clip(best_sc / 0.60, 0.0, 1.0))
            c_gain = float(np.clip(gain / 0.25, 0.0, 1.0))
            dist_from_field = float(np.hypot(best_dx - u0, best_dy - v0))
            c_field = float(np.exp(-(dist_from_field**2) / (2.0 * 2.0**2)))

            raw_conf = 0.40 + 0.35 * c_edge + 0.15 * c_gain + 0.10 * c_field
            conf = float(np.clip(raw_conf, 0.45, 0.95))

            if conf >= flag_threshold:
                shifted_3857 = translate(geom_3857, xoff=best_dx, yoff=best_dy)
                corrected_indices.append(len(results))
                shifted_geoms_3857.append(shifted_3857)

                results.append({
                    "plot_number": str(pn),
                    "status": "corrected",
                    "confidence": round(conf, 3),
                    "method_note": f"aligned dx={best_dx:.2f}m dy={best_dy:.2f}m (score={best_sc:.2f}, gain={gain:+.2f})",
                    "geometry": None,  # Will be populated after vector reprojection
                })
            else:
                results.append({
                    "plot_number": str(pn),
                    "status": "flagged",
                    "confidence": None,
                    "method_note": f"low confidence ({conf:.2f}); official retained",
                    "geometry": geom_4326,
                })

    # Reproject only the shifted corrected geometries back to EPSG:4326
    if corrected_indices:
        corr_series_3857 = gpd.GeoSeries(shifted_geoms_3857, crs="EPSG:3857")
        corr_series_4326 = corr_series_3857.to_crs("EPSG:4326")
        for list_idx, reprojected_geom in zip(corrected_indices, corr_series_4326):
            results[list_idx]["geometry"] = reprojected_geom

    gdf_4326 = gpd.GeoDataFrame(results, crs="EPSG:4326")
    gdf_4326 = gdf_4326.set_index("plot_number", drop=False)
    return gdf_4326
