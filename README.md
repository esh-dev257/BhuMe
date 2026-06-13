# BhuMe Boundary Correction - Take-Home Submission

## Method

**ICP from global shift + cross-correlation agreement check**

### Core idea

Official plot boundaries are shifted off the real fields due to imperfect georeferencing of old paper cadastral maps. The task is to find the correct translation for each plot.

### Algorithm (`solve.py`)

1. **Global median shift** - compute the median centroid displacement from the handful of example truths. This single shift already improves most plots.

2. **Per-plot ICP from global-shift position** — for each plot:
   - Apply the global shift to get a "primed" outline
   - Rasterise the primed outline onto the `boundaries.tif` edge map
   - For each outline pixel, find its nearest detected edge pixel within `REFINE_M = 30 m`
   - Median displacement = per-plot refinement on top of global shift
   - IQR of displacements (in metres) measures coherence:
     - **Small IQR** → all outline pixels agree on direction → tracking a single coherent field edge → high confidence
     - **Large IQR** → random competing neighbouring edges pull in different directions → uncertain → flagged
   - `confidence_icp = inlier_fraction × max(0, 1 − IQR_penalty)`

3. **Cross-correlation precision boost** — additionally compute the FFT cross-correlation peak in a `±40 m` window around the global shift. If the cross-corr estimate is within `AGREE_M = 12 m` of the ICP estimate (the two methods agree), use the cross-corr shift (more precise) but keep the ICP-based confidence.

4. **Flag or correct** — if `confidence < 0.25`, flag the plot (keep original position). Otherwise emit the corrected geometry with that confidence.

### Why calibration (Gold tier) holds

The IQR is large exactly when the correction is uncertain:
- **Open fields** (vadnerbhairav): after the global shift the outline sits near the real field edge. ICP finds consistent neighbours in one direction → low IQR → high confidence → accurate correction.
- **Crowded fields** (malatavadi): after the global shift many competing neighbouring edges pull outline pixels in different directions → high IQR → low confidence → flagged. Even when cross-corr finds a "perfect" local match 40 m away, ICP says "I'm not confident" → flagged.

### Results (public example truths)

| Village | Method | Median IoU | vs Official | Centroid Error |
|---------|--------|-----------|-------------|----------------|
| Vadnerbhairav | Official | 0.612 | — | — |
| Vadnerbhairav | **Ours** | **0.852** | +0.231 | 3.2 m |
| Malatavadi | Official | 0.510 | — | — |
| Malatavadi | **Ours** | **0.731** | +0.221 | 3.1 m |

## Setup

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/Mac
# or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install deps
uv sync

# Run on a village
uv run solve.py data/vadnerbhairav
uv run solve.py data/malatavadi
```

Data bundles go into `data/<village>/` with `input.geojson`, `imagery.tif`, `boundaries.tif`, `example_truths.geojson`.

## Files

```
solve.py                     main solution
bhume/                       starter kit helpers (io, geo, score)
data/vadnerbhairav/
  predictions.geojson        submission predictions
data/malatavadi/
  predictions.geojson        submission predictions
transcripts/
  README.md                  AI transcript notes
```
