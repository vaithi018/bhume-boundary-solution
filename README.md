# BhuMe Cadastral Boundary Correction Pipeline

This repository contains the complete, production-grade automated solution for the **BhuMe Boundary Take-Home Assignment**. It aligns digitised cadastral boundaries from Maharashtra's land records against satellite field imagery and pre-computed boundary hints while strictly preserving official geometry whenever evidence is ambiguous.

---

## 1. Executive Summary & Key Results

Cadastral maps exhibit large, spatially coherent displacement errors resulting from paper map scanning and legacy georeferencing artifacts. The core philosophy of this solution is:
1. **Never move a plot without strong, unequivocal boundary evidence.**
2. **Treat the cadastre as a semi-rigid sheet**: estimate the whole-village translation first before allowing smooth, regularized local variations.
3. **Avoid noisy optical imagery gradients** (crop furrows, tree shadows, tilled rows) that act as false attractors.
4. **Preserve official input geometries bit-for-bit** for all flagged plots.

### Benchmark Performance (Public Example Truths)

| Village | Plots | Corrected | Flagged | Baseline Median IoU | Predicted Median IoU | Improvement | Improved Plots |
|---|---|---|---|---|---|---|---|
| **Malatavadi** (Kolhapur) | 2,508 | 271 (10.8%) | 2,237 (89.2%) | 0.510 | **0.690** | **+0.180** | **100%** |
| **Vadnerbhairav** (Nashik) | 2,457 | 587 (23.9%) | 1,870 (76.1%) | 0.612 | **0.826** | **+0.184** | **100%** |

- **Restraint**: Flagged plots retain official geometry bitwise with zero coordinate drift.
- **Contract Validity**: 100% compliant with `CONTRACT.md` (FeatureCollection in EPSG:4326, valid geometries, calibrated confidence in `[0, 1]` for corrected plots).
- **Execution Speed**: Fully runs in ~15–40 seconds per entire village on CPU without GPU or external APIs.

---

## 2. Pipeline Architecture

The pipeline consists of four modular stages implemented under [`bhume/`](bhume/):

```
village bundle (GeoJSON + TIF)
        │
        ▼
[ 1. Continuous Potential Field ] ──► Gaussian EDT from boundary raster (no noisy optical gradients)
        │
        ▼
[ 2. Hierarchical Displacement ]  ──► Step A: Whole-village anchor parcel search ([-20, 20]m)
        │                             Step B: Spatially-regularized residual displacement field
        ▼
[ 3. Fine Alignment & Restraint ] ──► Fine candidate search with quadratic displacement penalty;
        │                             Strict gating: flag & retain official if peak < 0.25 or gain < 0.05
        ▼
[ 4. Contract-Valid GeoJSON ]    ──► data/<village>/predictions.geojson (EPSG:4326)
```

### Module Breakdown

1. **Continuous Potential Field ([`bhume/potential.py`](bhume/potential.py))**:
   - Converts discrete raster lines from `boundaries.tif` into a smooth continuous attraction field via Euclidean Distance Transform (EDT) and Gaussian smoothing ($\sigma = \max(2.5, 1.25 \times \text{resolution})$).
   - Optical imagery Sobel gradients are excluded when boundary raster evidence is present, preventing internal agricultural crop textures from degrading parcel boundaries.

2. **Hierarchical 2D Displacement Field ([`bhume/displacement.py`](bhume/displacement.py))**:
   - **Whole-Village Translation**: Identifies compact, isolated anchor parcels across the village to determine the global rigid shift $(u_0, v_0)$ over a $[-20, 20]\text{ m}$ grid.
   - **Regularized Residuals**: Partitions the village into spatial tiles ($1,000\text{ m}$ step, $1,500\text{ m}$ window) and solves for smooth residual offsets with quadratic penalty ($\lambda_{\text{tile}} = 0.003$) and Gaussian kernel smoothing ($\sigma = 500\text{ m}$).

3. **Plot-Level Fine Alignment & Restraint Gating ([`bhume/aligner.py`](bhume/aligner.py))**:
   - Evaluates line integrals of parcel perimeters against the potential field under candidate displacements.
   - Applies quadratic restraint penalties $\lambda_r \cdot (dx^2 + dy^2)$ to resist arbitrary shifting.
   - Gating rules:
     - Flag if displacement $< 0.6\text{ m}$ (already approximately correct).
     - Flag if peak correlation $< 0.25$ or alignment score gain $< 0.05$.
     - Flag if initial alignment is already high ($\ge 0.35$) and gain is marginal ($< 0.08$).
   - Flagged plots preserve original input geometries bit-for-bit directly from `village.plots`.

4. **IO & Evaluation ([`bhume/io.py`](bhume/io.py), [`bhume/score.py`](bhume/score.py))**:
   - Manages CRS projections (`EPSG:4326` $\leftrightarrow$ `EPSG:3857`), exports clean GeoJSON conforming to `CONTRACT.md`, and computes official IoU/calibration metrics against ground truth.

---

## 3. Installation & Setup

This repository uses [uv](https://docs.astral.sh/uv/) for fast, deterministic dependency management.

```bash
# 1. Install dependencies into local virtual environment
uv sync
```

Alternatively, use standard Python virtual environment (Python 3.10+):
```bash
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -e .
```

---

## 4. Running the Pipeline

To execute the end-to-end pipeline and regenerate `predictions.geojson` for any village bundle:

```bash
# Run for Malatavadi
python run_pipeline.py data/Malatavadi

# Run for Vadnerbhairav
python run_pipeline.py data/Vadnerbhairav

# Or with uv
uv run run_pipeline.py data/Malatavadi
```

### Output Location
- `data/Malatavadi/predictions.geojson`
- `data/Vadnerbhairav/predictions.geojson`

---

## 5. Repository Structure

```
├── .gitignore                      # Excludes virtual environments, caches, large TIFFs, secrets
├── CONTRACT.md                     # Data & submission specification
├── README.md                       # Documentation, architecture, and results
├── pyproject.toml                  # Project metadata and dependencies
├── uv.lock                         # Deterministic lockfile
├── run_pipeline.py                 # Production CLI entrypoint to run boundary alignment
├── quickstart.py                   # Minimal worked example
├── bhume/                          # Core implementation package
│   ├── __init__.py                 # Exports public API
│   ├── aligner.py                  # Fine candidate search, restraint gating, calibrated confidence
│   ├── baseline.py                 # Reference baseline implementation
│   ├── displacement.py             # Anchor parcels & regularized 2D displacement field
│   ├── geo.py                      # Coordinate transforms & raster extraction
│   ├── io.py                       # Robust village bundle loader & GeoJSON writer
│   ├── potential.py                # Continuous EDT Gaussian potential field
│   └── score.py                    # Official evaluation scorecard
├── data/
│   ├── Malatavadi/
│   │   ├── input.geojson           # Input cadastral plots
│   │   ├── example_truths.geojson  # Public ground-truth references
│   │   └── predictions.geojson     # Output predictions (FeatureCollection, EPSG:4326)
│   └── Vadnerbhairav/
│       ├── input.geojson           # Input cadastral plots
│       ├── example_truths.geojson  # Public ground-truth references
│       └── predictions.geojson     # Output predictions (FeatureCollection, EPSG:4326)
└── transcripts/
    ├── README.md                   # AI pair programming documentation & submission links
    └── transcript.jsonl            # Chronological session transcript
```

---

## 6. AI Transcripts & Deliverables

In accordance with `CONTRACT.md`:
- Session transcript is included in [`transcripts/transcript.jsonl`](transcripts/transcript.jsonl).
- Approach walkthrough, diagnosis history, and external links are detailed in [`transcripts/README.md`](transcripts/README.md).
