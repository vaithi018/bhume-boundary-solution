#!/usr/bin/env python3
"""
Full automated BhuMe boundary alignment pipeline.

Usage:
    python run_pipeline.py data/Malatavadi
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from bhume import (
    align_plots,
    build_potential_field,
    estimate_displacement_field,
    load,
    score,
    write_predictions,
)

DEFAULT_VILLAGE = 'data/Malatavadi'


def main(village_dir: str = DEFAULT_VILLAGE) -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    t_start = time.time()
    village_path = Path(village_dir)
    print(f'=== Starting BhuMe Alignment Pipeline for {village_path.name} ===', flush=True)

    # 1. Load Village
    village = load(village_path)
    n_plots = len(village.plots)
    n_truths = len(village.example_truths) if village.example_truths is not None else 0
    print(f'Loaded {village.slug}: {n_plots} plots, {n_truths} example truths', flush=True)

    # 2. Build Continuous Boundary Potential Field
    t0 = time.time()
    print('Building continuous boundary potential field from imagery and boundaries...', flush=True)
    potential = build_potential_field(village.boundaries_path, village.imagery_path)
    print(f'  Potential grid ready ({potential.grid.shape[1]}x{potential.grid.shape[0]}) in {time.time() - t0:.2f}s', flush=True)

    # 3. Estimate Adaptive 2D Displacement Field
    t0 = time.time()
    print('Estimating local adaptive displacement field with spatial regularization...', flush=True)
    field = estimate_displacement_field(village.plots.to_crs('EPSG:3857'), potential)
    n_active_tiles = int((field.tile_weights > 0.3).sum())
    print(f'  Displacement field fitted ({len(field.tile_centers)} tiles, {n_active_tiles} active displacement blocks) in {time.time() - t0:.2f}s', flush=True)

    # 4. Fine Plot Alignment, Restraint Gating, and Calibrated Confidence
    t0 = time.time()
    print('Performing plot-level boundary alignment with restraint gating...', flush=True)
    preds = align_plots(village, potential, field)
    n_corrected = int((preds['status'] == 'corrected').sum())
    n_flagged = int((preds['status'] == 'flagged').sum())
    print(f'  Alignment complete: {n_corrected} corrected ({n_corrected/n_plots:.1%}), {n_flagged} flagged ({n_flagged/n_plots:.1%}) in {time.time() - t0:.2f}s', flush=True)

    # 5. Export Contract-Valid predictions.geojson
    out_path = village.dir / 'predictions.geojson'
    written = write_predictions(out_path, preds)
    print(f'Wrote contract-valid predictions to {written} ({out_path.stat().st_size / 1024:.1f} KB)', flush=True)

    # 6. Evaluation against example truths
    if village.example_truths is not None and len(village.example_truths) > 0:
        print('\n' + '=' * 60, flush=True)
        print('SELF-SCORING REPORT (vs example truths)', flush=True)
        print('=' * 60, flush=True)
        scorecard = score(preds, village)
        print(str(scorecard), flush=True)

    print(f'\nTotal pipeline execution time: {time.time() - t_start:.2f}s', flush=True)


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VILLAGE
    main(target)
