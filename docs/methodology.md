# Methodology

This document details the scientific methodology, formulas, thresholds, and literature basis for each stage of the Terrain Analysis pipeline. Every formula and threshold below is reproduced exactly as implemented in `notebooks/terrain_analysis/PSR_Mapping.ipynb` and `src/terrain_analysis/`; nothing here has been altered during the notebook-to-package refactor. Full citations are in `docs/references.md`.

## 1. Horizon-Angle Modeling

**Reference:** Mazarico, Neumann, Smith, Zuber & Torrence (2011), eq. 1–2.

For each DEM pixel and a set of azimuth directions, the horizon angle is the maximum elevation angle at which terrain blocks the Sun along that direction. This pipeline computes it for **72 azimuth directions** (5° spacing) — a deliberate reduction from Mazarico et al.'s 720 directions (0.5° spacing), chosen as computationally feasible at 100 m/pixel resolution while keeping shadow-boundary error within roughly one pixel at crater scale.

For a ray cast at distance step `k` (in metres, `distances_m[k] = k × pixel_size`):

```
apparent_elevation = true_elevation(neighbor) − curvature_correction[k]
curvature_correction[k] = distances_m[k]² / (2 × R_MOON)
horizon_angle = max over k of: arctan2(apparent_elevation − elevation(pixel), distances_m[k])
```

where `R_MOON = 1,737,400 m`. The lunar curvature correction accounts for the fact that, at tens of kilometers, the Moon's curvature measurably lowers the apparent height of distant terrain.

**Search radius note:** the notebook cell that actually produced the saved `horizon_angles.npy` used `MAX_DIST_M = 50,000 m` (500 steps at 100 m/pixel), while an earlier configuration cell set `MAX_DIST_M = 100,000 m`. The pipeline (`horizon_angles.compute_horizon_angles`) defaults to 50,000 m to match the value that actually generated the pipeline's saved outputs — this discrepancy is preserved and documented rather than silently resolved (see `src/terrain_analysis/horizon_angles.py` docstring).

**Implementation:** `src/terrain_analysis/horizon_angles.py`, `compute_horizon_chunk` (Numba `@njit(parallel=True)`) and `compute_horizon_angles` (orchestration).

## 2. Solar Ephemeris

**Reference:** Astropy DE421 planetary ephemeris; IAU Working Group on Cartographic Coordinates and Rotational Elements (WGCCRE) Moon rotation model (Archinal et al., 2018).

The Sun's position relative to the Moon is computed via `astropy.coordinates.get_body_barycentric_posvel`, then rotated from the ICRS frame into the Moon-Centered Moon-Fixed (MCMF) frame using a manually-implemented IAU rotation (`build_icrs_to_mcmf_rotation`) — the same rotation model underlying the SPICE PCK kernels (`pck00010.tpc`/`pck00011.tpc`) referenced in the LIRAF proposal. The Sun's position is then projected onto a local up/north/east frame at the observation point (Faustini crater, 87.3°S, 84.5°E in this implementation) to yield azimuth and elevation.

The pipeline samples every 6 hours over 1 simulated year (1,461 timesteps) by default.

**Validation performed:** azimuth spans the full 0–360°, elevation stays within the physically expected range for the Moon's small axial tilt (~1.54°), and the fraction of time the Sun is above the local horizon is reported as a sanity check.

**Implementation:** `src/terrain_analysis/illumination.py`, `build_icrs_to_mcmf_rotation` and `compute_solar_ephemeris`.

## 3. Illumination Fraction & PSR Mask

**Reference:** Mazarico et al. (2011), eq. 3.

For pixel `(r, c)` at timestep `t`, with the Sun at azimuth bin `k` and elevation `sun_elevation[t]`:

```
illuminated(r, c, t) = sun_elevation[t] > horizon_angles[k, r, c]
illumination_fraction(r, c) = (number of illuminated timesteps) / (total timesteps)
```

A pixel is classified as a **Permanently Shadowed Region (PSR)** if its illumination fraction is exactly `0.0` across the full simulated period — i.e. the Sun never rises above that pixel's horizon at any sampled time, for any azimuth the Sun passes through.

**Implementation:** `src/terrain_analysis/illumination.py`, `compute_illumination_and_psr`.

## 4. Candidate Ice-Trap Detection

**References:** Mazarico et al. (2011) [PSR definition], Paige et al. (2010) [thermal cold-trap criteria], Hayne et al. (2015) [cold-trap modeling], Rubanenko et al. (2019) [micro cold-trap theory], Watson et al. (1961) [classical cold-trap theory].

Within the PSR mask, four 0–1 normalized scoring layers are combined into a weighted composite:

| Layer | Formula | Weight |
|---|---|---|
| Illumination score | `1 − illumination_fraction` | 0.30 |
| Local depression score | pixel elevation vs. 500 m neighborhood mean, percentile-normalized | 0.25 |
| Slope score | `clip(1 − slope_deg/30, 0, 1)` (Sobel-based slope) | 0.20 |
| Rim-depth score | depth below the max elevation within a 20 km radius (regional cold-trap proxy), percentile-normalized | 0.25 |

```
composite = 0.30·illum_score + 0.25·depression_score + 0.20·slope_score + 0.25·rim_depth_score
composite[outside PSR] = 0.0
```

Candidate regions are pixels in the top 20% of in-PSR composite scores; high-confidence regions are the top 10%. Connected components (8-connectivity) smaller than **0.5 km²** are discarded. Remaining regions are ranked by mean composite score.

**Implementation:** `src/terrain_analysis/ice_trap_detection.py`, `detect_candidate_ice_traps`.

## 5. Landing-Site Selection

**References:** Arvidson et al. (2002) [slope hazard limit], Mazarico et al. (2011) [illumination threshold], Rubanenko et al. (2019) [traverse-distance criterion], Kreslavsky & Head (2000) [roughness metric].

**Hard exclusion criteria** (a pixel must satisfy all of these to be considered landable):

| Criterion | Threshold |
|---|---|
| Slope | `< 10°` (Sobel-based slope) |
| Illumination fraction | `> 20%` |
| Roughness (local slope std-dev, 500 m window) | `< 5°` |
| Not a deep PSR interior | excludes pixels where `PSR AND illumination < 5%` |

**Composite landing score** (weighted sum of percentile-normalized sub-scores):

| Sub-score | Formula | Weight |
|---|---|---|
| Slope score | `norm01(30 − slope_deg)` | 0.35 |
| Illumination score | `norm01(illumination_fraction)` | 0.25 |
| Ice-proximity score | `norm01(50 − distance_to_ice_km)` | 0.25 |
| Roughness score | `norm01(10 − roughness_deg)` | 0.15 |

Sites are formed from the top 30% of landable-pixel scores, connected-component labeled (8-connectivity), with clusters below **0.1 km²** discarded. Sites are ranked by mean score, tie-broken by proximity to the nearest high-confidence ice trap. The top-ranked site receives an auto-generated scientific justification string built from its own computed statistics (slope, illumination, distance to ice).

**Implementation:** `src/terrain_analysis/landing_site_scoring.py`, `rank_landing_sites`.

## 6. Standalone Terrain Products

Independent of the PSR/ice-trap/landing-site pipeline, two standard terrain-derivative rasters are produced directly from the DEM:

- **Slope map** — `np.gradient`-based slope in degrees (note: numerically distinct from the Sobel-based slope used in sections 4–5 above; see `docs/architecture.md` for why these are kept separate).
- **Hillshade** — standard analytical hillshade formula, sun azimuth 315°, altitude 45°.

**Implementation:** `src/terrain_analysis/terrain_processing.py`, `compute_slope_gradient`/`save_slope_map` and `compute_hillshade`/`save_hillshade`.

## Assumptions Carried Through the Pipeline

- DEM resolution: 100 m/pixel throughout.
- Nodata pixels are filled with the DEM mean for operations requiring a complete array (e.g. gradient computation), while the original NaN mask is preserved separately wherever the distinction matters.
- Faustini crater coordinates (87.3°S, 84.5°E) are used as the fixed observation point for the solar-ephemeris local frame; this is specific to this DEM's extent, not a general-purpose default.
- All weighted scoring models (ice-trap composite, landing-site composite) are weighted additive combinations, not learned or fitted models — weights were fixed based on cited literature thresholds and standard ESA/ISRO site-selection practice, as stated in the original notebook.