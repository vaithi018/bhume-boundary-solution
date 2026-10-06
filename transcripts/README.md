# AI Transcripts & Methodology Walkthrough

This directory documents the AI-assisted engineering and pair programming workflow used to design, diagnose, implement, and validate the BhuMe cadastral boundary alignment pipeline.

Per the assignment specification in `CONTRACT.md`:
> *"You also send a 5-minute video walking through your approach to the problem, and your AI transcripts (no written report). We expect AI used two ways, to understand the problem (web chats) and to build the solution (coding tools); keep them in a `/transcripts` folder in your repo, with any web-chat share links listed in `transcripts/README.md`."*

---

## 1. Submission Links

- **5-Minute Approach Walkthrough Video**: `[Insert link here, e.g., Loom / YouTube / Google Drive]`
- **Web-Chat Share Links (Problem Understanding / Initial Framing)**:
  - `[Insert web-chat link 1]`
  - `[Insert web-chat link 2]`

---

## 2. Included Coding Session Transcripts

- [`transcript.jsonl`](transcript.jsonl): The complete chronological transcript of the coding assistant session that diagnosed the initial baseline degradation, eliminated imagery gradient contamination, implemented rigid-sheet anchor translation and spatially-regularized residual displacement fields, enforced quadratic restraint gating, and validated contract compliance across both Malatavadi and Vadnerbhairav.

---

## 3. Key Reasoning & Engineering Journey

### Phase 1: Diagnosing Pipeline Degradation
- **Problem**: The initial boundary alignment implementation severely degraded cadastral accuracy, dropping median IoU on Malatavadi from 0.510 down to 0.179 (improvement: -0.269).
- **Diagnosis**:
  1. Optical imagery Sobel gradients were contaminating the potential field with internal agricultural field textures (crop rows, tree canopies, contour bunds), creating false local minima every 3–10 metres.
  2. Independent tile displacement optimization lacked spatial coherence (64.2% of adjacent tiles diverged by $>5\text{ m}$), causing topological tearing across adjacent parcels.
  3. Unconstrained movement shifted plots that were already near their true positions.
  4. Coordinate round-tripping (`EPSG:4326` $\to$ `EPSG:3857` $\to$ `EPSG:4326`) altered flagged plots bitwise.

### Phase 2: Simplified, Robust Algorithmic Solution
- **Continuous Potential Field (`bhume/potential.py`)**:
  - Replaced noisy imagery Sobel gradients with a clean, continuous Gaussian Euclidean Distance Transform (EDT) potential derived from `boundaries.tif` ($\sigma = \max(2.5, 1.25 \times \text{res})$).
- **Hierarchical Displacement Estimation (`bhume/displacement.py`)**:
  - Automatically identifies high-confidence anchor parcels to determine a robust global rigid-sheet translation $(u_0, v_0)$ over $[-20, 20]\text{ m}$.
  - Fits a smooth low-rank residual displacement field around $(u_0, v_0)$ across spatial tiles ($1,000\text{ m}$ step, $1,500\text{ m}$ window) with quadratic regularization ($\lambda = 0.003$) and Gaussian kernel smoothing ($\sigma = 500\text{ m}$).
- **Restraint Gating & Bitwise Preservation (`bhume/aligner.py`)**:
  - Evaluates fine candidate displacements using potential line integrals with quadratic displacement penalties.
  - Plots are flagged and strictly preserved bit-for-bit with the official geometry if confidence or boundary evidence is insufficient ($\Delta \text{score} < 0.05$ or initial alignment already high).

### Phase 3: Verification on Multiple Villages
- **Malatavadi (2,508 plots)**:
  - Median IoU: **0.690** vs Official Baseline **0.510** (+0.180 improvement).
  - Flags 89.2% of plots where signal is ambiguous, achieving 100% improvement on corrected plots.
- **Vadnerbhairav (2,457 plots)**:
  - Median IoU: **0.826** vs Official Baseline **0.612** (+0.184 improvement).
  - Generalizes cleanly without village-specific tuning or hardcoding.
