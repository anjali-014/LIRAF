# """
# Horizon-angle computation for the Faustini DEM.

# Extracted from PSR_Mapping.ipynb, "CELL 6 (OPTIMIZED) — Horizon Angles
# with Curvature Correction". Implements Mazarico et al. (2011), eq. 1-2:
# for every DEM pixel and a set of azimuth directions, computes the
# elevation angle of the local horizon, corrected for lunar curvature.
# This is the geometric input required by illumination.py to later
# determine, at each simulated timestep, whether the Sun is visible above
# each pixel's local horizon.

# No algorithm, threshold, or parameter has been changed from the original
# notebook cell. The Numba-accelerated inner loop (`compute_horizon_chunk`)
# is reproduced verbatim; only the surrounding setup/chunking/save logic has
# been wrapped into a single callable function with explicit parameters in
# place of notebook globals.
# """

# import os
# import time
# from typing import Tuple

# import numpy as np
# import rasterio
# from numba import njit, prange
# import matplotlib.pyplot as plt

# from .utils import load_dem, chunk_path, ensure_dir

# # Lunar radius (metres), Mazarico et al. (2011) eq. 2 curvature correction.
# # Reproduced from PSR_Mapping.ipynb cell 10.
# R_MOON_M = 1_737_400.0


# @njit(parallel=True)
# def compute_horizon_chunk(
#     dem_chunk: np.ndarray,
#     dem_full: np.ndarray,
#     chunk_start_row: int,
#     drows: np.ndarray,
#     dcols: np.ndarray,
#     distances_m: np.ndarray,
#     curv_corr: np.ndarray,
#     pixel_size: float,
#     max_steps: int,
#     n_az: int,
#     full_rows: int,
#     full_cols: int,
# ) -> np.ndarray:
#     """
#     Compute horizon angles for a chunk of DEM rows.

#     Reproduced verbatim from PSR_Mapping.ipynb cell 10. For each pixel
#     ``(r, c)`` in the chunk and each azimuth direction ``i``, this scans
#     outward in steps ``k = 1..max_steps`` along direction
#     ``(drows[i], dcols[i])``, computes the neighbor elevation corrected
#     for lunar curvature (``elev_apparent = dem[nr, nc] - curv_corr[k]``),
#     and takes the maximum horizon angle
#     (``arctan2(elev_apparent - dem[r, c], distances_m[k])``) over all
#     steps.

#     Parameters
#     ----------
#     dem_chunk : np.ndarray
#         DEM rows for this chunk only, shape ``(chunk_rows, full_cols)``.
#     dem_full : np.ndarray
#         Full DEM array, shape ``(full_rows, full_cols)`` — needed because
#         rays cast from this chunk may look into neighboring chunks.
#     chunk_start_row : int
#         Row offset of this chunk within the full DEM.
#     drows, dcols : np.ndarray
#         Per-azimuth unit step vectors in pixel-row/column space.
#     distances_m : np.ndarray
#         Distance in metres at each ray step, shape ``(max_steps,)``.
#     curv_corr : np.ndarray
#         Lunar-curvature correction (metres) at each ray step, shape
#         ``(max_steps,)``, per Mazarico et al. (2011) eq. 2.
#     pixel_size : float
#         DEM pixel size in metres.
#     max_steps : int
#         Maximum number of ray steps to search for the horizon.
#     n_az : int
#         Number of azimuth directions.
#     full_rows, full_cols : int
#         Shape of the full DEM.

#     Returns
#     -------
#     np.ndarray
#         Horizon angles in radians, shape ``(n_az, chunk_rows, full_cols)``.
#     """
#     chunk_rows_n = dem_chunk.shape[0]
#     horizon = np.full((n_az, chunk_rows_n, full_cols),
#                       -np.pi / 2, dtype=np.float32)

#     for local_r in prange(chunk_rows_n):
#         global_r = chunk_start_row + local_r
#         for c in range(full_cols):
#             base_elev = dem_chunk[local_r, c]

#             for i in range(n_az):
#                 dr = drows[i]
#                 dc = dcols[i]
#                 max_angle = -np.pi / 2

#                 for k in range(max_steps):
#                     nr = global_r + dr * (k + 1)
#                     nc = c + dc * (k + 1)

#                     ri = int(nr + 0.5)
#                     ci = int(nc + 0.5)

#                     if ri < 0 or ri >= full_rows:
#                         break
#                     if ci < 0 or ci >= full_cols:
#                         break

#                     elev_apparent = dem_full[ri, ci] - curv_corr[k]

#                     angle = np.arctan2(
#                         elev_apparent - base_elev,
#                         distances_m[k]
#                     )

#                     if angle > max_angle:
#                         max_angle = angle

#                 horizon[i, local_r, c] = max_angle

#     return horizon


# def compute_horizon_angles(
#     dem_path: str,
#     output_dir: str,
#     n_azimuths: int = 72,
#     max_dist_m: float = 50_000,
#     pixel_size_m: float = 100,
#     chunk_rows: int = 100,
#     make_diagnostic_plot: bool = True,
# ) -> np.ndarray:
#     """
#     Compute and save horizon angles for every pixel in a DEM.

#     Reproduces the orchestration logic of PSR_Mapping.ipynb cell 10:
#     loads the DEM, precomputes azimuth unit vectors and lunar-curvature
#     correction, processes the DEM in resumable row-chunks (saving each
#     chunk to disk so an interrupted run can resume), merges the chunks,
#     saves the final ``horizon_angles.npy``, and optionally produces a
#     diagnostic plot.

#     Note on parameter defaults: the original notebook set ``N_AZIMUTHS
#     = 72`` in two places with slightly different values for
#     ``MAX_DIST_M`` (100,000 m in the Section 0 config cell vs.
#     50,000 m in this cell's own local config). The defaults here
#     reproduce the value actually used by this cell (50,000 m), since
#     that is what generated the saved ``horizon_angles.npy`` output this
#     pipeline depends on.

#     Parameters
#     ----------
#     dem_path : str
#         Path to the DEM GeoTIFF.
#     output_dir : str
#         Directory to save ``horizon_angles.npy``, chunk files, and the
#         diagnostic plot into.
#     n_azimuths : int, default 72
#         Number of azimuth directions (5° spacing at the default value).
#         Mazarico et al. (2011) used 720 (0.5° spacing); 72 was chosen in
#         the original notebook as computationally feasible at 100 m/pixel
#         while keeping shadow-boundary error under one pixel at crater
#         scale (per the notebook's own justification print-out).
#     max_dist_m : float, default 50_000
#         Maximum horizon search distance in metres.
#     pixel_size_m : float, default 100
#         DEM pixel size in metres.
#     chunk_rows : int, default 100
#         Number of DEM rows processed per resumable chunk.
#     make_diagnostic_plot : bool, default True
#         Whether to generate and save the diagnostic horizon-map plot.

#     Returns
#     -------
#     np.ndarray
#         Horizon angles in radians, shape ``(n_azimuths, rows, cols)``.
#         Also saved to ``{output_dir}/horizon_angles.npy``.
#     """
#     max_steps = int(max_dist_m / pixel_size_m)
#     chunk_dir = os.path.join(output_dir, "horizon_chunks")
#     horizon_path = os.path.join(output_dir, "horizon_angles.npy")
#     ensure_dir(chunk_dir)

#     # ── Load DEM ──────────────────────────────────────────────────
#     dem_data = load_dem(dem_path)
#     dem = dem_data["dem"]
#     rows, cols = dem.shape
#     dem_filled = np.where(np.isnan(dem), np.nanmean(dem), dem)

#     print(f"DEM loaded: {rows} x {cols}")
#     print(f"Total pixels : {rows * cols:,}")
#     print(f"Operations   : {rows * cols * n_azimuths * max_steps / 1e9:.1f} billion")

#     # ── Azimuth unit vectors (precomputed once) ──────────────────
#     azimuths_deg = np.linspace(0, 360, n_azimuths, endpoint=False)
#     azimuths_rad = np.radians(azimuths_deg)

#     # drow/dcol in pixel coords: az=0 -> North, az=90 -> East
#     drows = (-np.cos(azimuths_rad)).astype(np.float32)
#     dcols = (np.sin(azimuths_rad)).astype(np.float32)

#     # Precompute distance array (same for every pixel, every azimuth)
#     step_indices = np.arange(1, max_steps + 1, dtype=np.float32)
#     distances_m = step_indices * pixel_size_m  # shape (max_steps,)

#     # Curvature correction per step (Mazarico 2011 eq. 2)
#     # apparent_elev = true_elev - d^2 / (2R)
#     curv_corr = (distances_m ** 2) / (2.0 * R_MOON_M)  # shape (max_steps,)

#     # ── Resume logic: which chunks are done? ─────────────────────
#     n_chunks = (rows + chunk_rows - 1) // chunk_rows

#     def _chunk_path(ch_idx: int) -> str:
#         return chunk_path(chunk_dir, "chunk_", ch_idx)

#     done = [os.path.exists(_chunk_path(i)) for i in range(n_chunks)]
#     print(f"\nChunks total    : {n_chunks}")
#     print(f"Already done    : {sum(done)}")
#     print(f"Remaining       : {n_chunks - sum(done)}")

#     # ── Main loop ─────────────────────────────────────────────────
#     total_start = time.time()

#     for ch in range(n_chunks):
#         if done[ch]:
#             print(f"  Chunk {ch + 1:03d}/{n_chunks} — skipped (already saved)")
#             continue

#         r_start = ch * chunk_rows
#         r_end = min(r_start + chunk_rows, rows)
#         dem_chunk = dem_filled[r_start:r_end, :]

#         t0 = time.time()
#         hz = compute_horizon_chunk(
#             dem_chunk, dem_filled,
#             r_start,
#             drows, dcols,
#             distances_m, curv_corr,
#             np.float32(pixel_size_m), max_steps, n_azimuths,
#             rows, cols,
#         )

#         np.save(_chunk_path(ch), hz)
#         elapsed = time.time() - t0
#         done_count = sum(os.path.exists(_chunk_path(i)) for i in range(n_chunks))
#         remaining_chunks = n_chunks - done_count
#         eta = elapsed * remaining_chunks
#         print(f"  Chunk {ch + 1:03d}/{n_chunks} "
#               f"rows {r_start}-{r_end} | "
#               f"{elapsed:.1f}s | "
#               f"ETA {eta / 60:.1f} min")

#     # ── Merge chunks ──────────────────────────────────────────────
#     print("\nMerging chunks into final array...")
#     horizon_angles = np.zeros((n_azimuths, rows, cols), dtype=np.float32)

#     for ch in range(n_chunks):
#         r_start = ch * chunk_rows
#         r_end = min(r_start + chunk_rows, rows)
#         horizon_angles[:, r_start:r_end, :] = np.load(_chunk_path(ch))

#     np.save(horizon_path, horizon_angles)
#     mem_mb = horizon_angles.nbytes / 1e6
#     print(f"Saved: {horizon_path}")
#     print(f"   Shape : {horizon_angles.shape}")
#     print(f"   Size  : {mem_mb:.0f} MB")

#     # ── Diagnostic plot ───────────────────────────────────────────
#     if make_diagnostic_plot:
#         max_hz_deg = np.degrees(np.max(horizon_angles, axis=0))
#         max_hz_deg[np.isnan(dem)] = np.nan

#         fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0a0a0a')
#         for ax in axes:
#             ax.set_facecolor('#0a0a0a')

#         im0 = axes[0].imshow(dem, cmap='gray')
#         plt.colorbar(im0, ax=axes[0]).set_label('Elevation (m)', color='white')
#         axes[0].set_title('DEM — Faustini region', color='white')
#         axes[0].tick_params(colors='white')

#         im1 = axes[1].imshow(max_hz_deg, cmap='inferno', vmin=0, vmax=20)
#         plt.colorbar(im1, ax=axes[1]).set_label('Max horizon angle (deg)', color='white')
#         axes[1].set_title(
#             'Horizon Map (curvature corrected)\nBright = deep walls = likely PSR',
#             color='white')
#         axes[1].tick_params(colors='white')

#         plt.suptitle('Horizon Angles  [Mazarico 2011 + curvature]',
#                      color='white', fontsize=13)
#         plt.tight_layout()
#         out_png = os.path.join(output_dir, "06_horizon_diagnostic.png")
#         plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
#         plt.show()
#         print(f"Diagnostic saved: {out_png}")

#     total_min = (time.time() - total_start) / 60
#     print(f"\nTotal wall time: {total_min:.1f} min")

#     return horizon_angles