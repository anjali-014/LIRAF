# Quick Start

This walkthrough reproduces the Terrain Analysis pipeline end-to-end, stage by stage, using the extracted `src/terrain_analysis/` package. It assumes you have already obtained the DEM — see [`docs/datasets.md`](../docs/datasets.md) for source and download instructions, since it is not committed to this repository.

## 1. Install dependencies

```bash
git clone https://github.com/anjali-014/LIRAF.git
cd LIRAF
pip install -r requirements.txt
```

## 2. Set up paths

```python
dem_path = "faustini_test_dem_100m.tif"   # place your downloaded DEM here
output_dir = "PSR_outputs"                 # all pipeline outputs will be written here

import os
os.makedirs(output_dir, exist_ok=True)
```

## 3. Run each stage

### Stage 1 — Horizon angles

```python
from terrain_analysis import horizon_angles

horizon = horizon_angles.compute_horizon_angles(dem_path, output_dir)
```

**Expected runtime:** approximately 40–60 minutes on a free-tier Colab CPU runtime, per the original notebook's own estimate (Numba-JIT accelerated). This is the slowest stage in the pipeline; it is chunked and resumable, so an interrupted run can be restarted without recomputing already-saved row chunks.

### Stage 2 — Solar ephemeris

```python
from terrain_analysis import illumination

sun_positions, sun_times = illumination.compute_solar_ephemeris(output_dir)
```

**Expected runtime:** a few minutes (1,461 timesteps by default, batched).

### Stage 3 — Illumination fraction & PSR mask

```python
illum_frac, psr_mask = illumination.compute_illumination_and_psr(
    horizon_path=f"{output_dir}/horizon_angles.npy",
    sun_positions_path=f"{output_dir}/sun_positions.npy",
    output_dir=output_dir,
    dem_path=dem_path,   # optional, only needed for diagnostic plots
)
```

### Stage 4 — Candidate ice-trap detection

```python
from terrain_analysis import ice_trap_detection

result = ice_trap_detection.detect_candidate_ice_traps(
    dem_path=dem_path,
    psr_mask_path=f"{output_dir}/psr_mask.npy",
    illumination_path=f"{output_dir}/illumination_fraction.npy",
    output_dir=output_dir,
)
print(f"Found {len(result['regions'])} candidate ice-trap regions")
```

### Stage 5 — Landing-site selection

```python
from terrain_analysis import landing_site_scoring

sites_result = landing_site_scoring.rank_landing_sites(
    dem_path=dem_path,
    psr_mask_path=f"{output_dir}/psr_mask.npy",
    illumination_path=f"{output_dir}/illumination_fraction.npy",
    ice_trap_path=f"{output_dir}/candidate_ice_traps.npy",
    output_dir=output_dir,
)
print("Best site:", sites_result["best_site"].get("centroid_lat"),
      sites_result["best_site"].get("centroid_lon"))
```

### Stage 6 — Standalone terrain products (independent of stages 1–5)

```python
from terrain_analysis import terrain_processing

terrain_processing.save_slope_map(dem_path, f"{output_dir}/slope_map.tif")
terrain_processing.save_hillshade(dem_path, f"{output_dir}/hillshade.tif")
terrain_processing.print_dem_reference_info(dem_path)
```

## 4. Reproduce a single stage only

Every function above takes the *file paths* of its required inputs, not in-memory objects from a prior stage — so you can re-run any single stage independently, as long as its required upstream `.npy` files already exist in `output_dir`. This matches the original notebook's own file-based staging (see [`docs/architecture.md`](../docs/architecture.md)).

For example, to re-run only the landing-site ranking after tweaking a threshold, you don't need to recompute horizon angles or illumination — just re-run Stage 5 directly, pointing at the existing `psr_mask.npy`, `illumination_fraction.npy`, and `candidate_ice_traps.npy`.

## 5. Running the notebook instead

If you prefer the full narrative walkthrough with inline figures, open `notebooks/terrain_analysis/PSR_Mapping.ipynb` directly — it runs the same underlying logic as the `src/` package (in fact, the `src/` package was extracted from it), with Markdown documentation at each stage explaining the objective, methodology, inputs, and outputs.

## Troubleshooting

- **`FileNotFoundError` on the DEM path** — you need to download the DEM yourself; see [`docs/datasets.md`](../docs/datasets.md).
- **Horizon-angle computation seems stuck** — this is the expected ~40–60 minute stage; check `output_dir/horizon_chunks/` for incrementally-saved chunk files to confirm progress.
- **`ImportError` for `jplephem`** — required by Astropy's ephemeris functions in Stage 2; ensure `pip install -r requirements.txt` completed successfully.