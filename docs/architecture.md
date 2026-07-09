# Terrain Analysis Architecture

This document describes the architecture of the Terrain Analysis module within the Lunar Ice Resource Assessment Framework (LIRAF). It explains how data flows through the processing pipeline, the responsibilities of each Python module, and how the terrain-analysis subsystem integrates with the broader multi-module hackathon project.

## Processing Pipeline

```
faustini_test_dem_100m.tif (LOLA DEM, external — see datasets.md)
        │
        ▼
┌───────────────────────┐
│ horizon_angles.py      │  compute_horizon_angles()
│                        │  → horizon_angles.npy  (72, rows, cols)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ illumination.py        │  compute_solar_ephemeris()
│                        │  → sun_positions.npy, sun_times.npy
│                        │
│                        │  compute_illumination_and_psr()
│                        │  → illumination_fraction.npy, psr_mask.npy
│                        │
│                        │  visualize_psr_mask(), compute_regional_psr_area()
└───────────┬───────────┘
            │
            ├─────────────────────────────┐
            ▼                             ▼
┌───────────────────────┐     ┌───────────────────────┐
│ ice_trap_detection.py  │     │ terrain_processing.py  │
│                        │     │                        │
│ detect_candidate_      │     │ save_slope_map()        │
│ ice_traps()            │     │ save_hillshade()        │
│ → candidate_ice_       │     │ export_combined_        │
│   traps.npy,           │     │   dataset()             │
│   *_catalogue.json     │     │ print_dem_reference_    │
└───────────┬───────────┘     │   info()                │
            │                  └───────────────────────┘
            ▼
┌───────────────────────┐
│ landing_site_scoring.py│  rank_landing_sites()
│                        │  → landing_sites.npy,
│                        │    landing_site_catalogue.json,
│                        │    best_landing_site.json
└───────────────────────┘

┌───────────────────────┐
│ utils.py               │  load_dem(), save_raster(),
│ (used by all modules)  │  chunk_path(), ensure_dir()
└───────────────────────┘
```

## Module Responsibilities

| Module | Responsibility | Depends on |
|---|---|---|
| `utils.py` | Generic I/O: raster loading/saving, chunk-file paths, directory creation. No science logic. | — |
| `horizon_angles.py` | Computes the horizon-angle geometry needed to determine solar visibility at every pixel, for every azimuth direction. Numba-accelerated. | `utils` |
| `illumination.py` | Computes the Sun's position over time (solar ephemeris) and combines it with horizon angles to derive annual illumination fraction and the boolean PSR mask. | `utils` |
| `ice_trap_detection.py` | Scores and ranks candidate ice-trapping sub-regions within the PSR, using illumination, topographic depression, slope, and rim-depth criteria. | `utils` |
| `landing_site_scoring.py` | Applies hard safety constraints and a weighted composite score to rank candidate landing sites, balancing safety against ice-trap proximity. | `utils` |
| `terrain_processing.py` | Standalone terrain products (slope map, hillshade) and tabular data export, independent of the PSR/ice-trap pipeline. | `utils` |

## Design Decisions

**File-based staging, not in-memory coupling.** Every stage after DEM loading reads its inputs from `.npy`/`.tif` files written by the previous stage, exactly mirroring how the original research notebook was structured (each notebook cell reloaded its inputs from disk rather than relying on in-memory variables from earlier cells). This was a deliberate choice during the `src/` extraction: it preserves the notebook's actual behavior rather than introducing a new in-memory architecture that could change how partial runs, resumption, or debugging behave.

**Two independently-computed slope rasters.** `ice_trap_detection.compute_slope_sobel` and `landing_site_scoring`'s inline slope computation use a Sobel-operator gradient, while `terrain_processing.compute_slope_gradient` uses `np.gradient`. These are retained as genuinely separate computations (see `docs/methodology.md` for details) because the original notebook computed them independently in different cells for different purposes — merging them would silently change either the ice-trap/landing-site scoring or the standalone terrain output.

**Two independently-defined plotting-style helpers.** `ice_trap_detection.label_ax` and `landing_site_scoring.style_ax` serve the same conceptual purpose (dark-theme axis styling) but were written separately in the source notebook with different implementation details. They are kept as two functions rather than unified into one shared helper.

## Where This Fits in LIRAF

LIRAF as a whole (per the BAH 2026 proposal) is a four-module system:

```
Radar Processing ──┐
                    │
Terrain Analysis ───┼──► Data Fusion (GMM) ──► Mission Planning (landing site
     (This Repository) │                        ranking, A*/RRT rover path,
                    │                            ice volume estimate)
OHRC Analysis ──────┘
```

This repository implements only the **Terrain Analysis** branch. Its outputs (`psr_mask.npy`, `illumination_fraction.npy`, and the candidate ice-trap/landing-site catalogues) are the inputs the team's Data Fusion and Mission Planning modules consume, per the proposal's architecture diagram — but the fusion/rover-planning code itself belongs to other teammates and is not included here. See `docs/team_contributions.md`.

This separation keeps the repository focused on the verified terrain-analysis work while maintaining clear interfaces for integration with the remaining LIRAF modules.