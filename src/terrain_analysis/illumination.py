"""
Solar ephemeris, illumination fraction, and Permanently Shadowed Region
(PSR) mapping for Faustini crater.

Extracted from PSR_Mapping.ipynb:
  - "CELL 7 — Solar Ephemeris for Faustini PSR Mapping" (cell 21)
  - "CELL 8 — Illumination Fraction + PSR Mask" (cell 23)
  - the PSR-mask visualization cell (cell 26)
  - the Faustini-specific regional PSR area calculation (cell 27)

Methodology: Mazarico, E., Neumann, G. A., Smith, D. E., Zuber, M. T., &
Torrence, M. H. (2011). Illumination Conditions of the Lunar Polar
Regions Using LOLA Topography.

No algorithm, threshold, or parameter has been changed from the original
notebook cells. Surrounding setup/save/plotting logic has been wrapped
into callable functions with explicit parameters in place of notebook
globals.
"""

import os
import time
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
import rasterio

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import (
    solar_system_ephemeris,
    get_body_barycentric_posvel,
    GCRS, ICRS,
    SkyCoord,
)

from .utils import ensure_dir, chunk_path

# Lunar radius (metres). Same physical constant as horizon_angles.R_MOON_M,
# reproduced locally here as it was in the original notebook (cell 21
# redefines it independently of cell 10/6's copy).
R_MOON_M = 1_737_400.0


def build_icrs_to_mcmf_rotation(t_astropy: Time) -> np.ndarray:
    """
    Build the rotation matrix from ICRS to the Moon-Centred Moon-Fixed
    (MCMF) frame at a given epoch.

    Reproduced verbatim from PSR_Mapping.ipynb cell 21. Uses the IAU
    WGCCRE 2015 Moon rotation model (Archinal et al. 2018, Celestial
    Mechanics and Dynamical Astronomy) — the same rotation model used in
    the SPICE PCK files (pck00010.tpc / pck00011.tpc) cited by
    Mazarico et al. (2011).

    Parameters
    ----------
    t_astropy : astropy.time.Time
        A single epoch to compute the rotation for.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix ``R`` such that ``v_mcmf = R @ v_icrs``.
    """
    # Days and centuries from J2000.0
    d = t_astropy.tdb.jd - 2451545.0  # Julian days from J2000
    T = d / 36525.0  # Julian centuries

    # Pole direction in ICRS (right ascension, declination)
    alpha0_deg = 269.9949 + 0.0031 * T  # degrees
    delta0_deg = 66.5392 + 0.0130 * T  # degrees
    W_deg = 38.3213 + 13.17635815 * d  # prime meridian angle

    alpha0 = np.radians(alpha0_deg)
    delta0 = np.radians(delta0_deg)
    W = np.radians(W_deg % 360.0)

    # Step 1: Build Moon pole unit vector in ICRS
    pole = np.array([
        np.cos(delta0) * np.cos(alpha0),
        np.cos(delta0) * np.sin(alpha0),
        np.sin(delta0)
    ])

    # Step 2: Build MCMF x-axis (prime meridian direction)
    icrs_z = np.array([0.0, 0.0, 1.0])
    node = np.cross(icrs_z, pole)
    node /= np.linalg.norm(node)

    # Rotate 'node' around 'pole' by angle W -> MCMF x-axis (Rodrigues formula)
    k = pole
    cW = np.cos(W)
    sW = np.sin(W)
    x_mcmf = (node * cW
              + np.cross(k, node) * sW
              + k * np.dot(k, node) * (1 - cW))
    x_mcmf /= np.linalg.norm(x_mcmf)

    # MCMF y-axis = pole x x_mcmf
    y_mcmf = np.cross(pole, x_mcmf)
    y_mcmf /= np.linalg.norm(y_mcmf)

    R = np.stack([x_mcmf, y_mcmf, pole], axis=0)  # shape (3,3)
    return R


def compute_solar_ephemeris(
    output_dir: str,
    sim_years: float = 1,
    sun_timestep_h: float = 6,
    faustini_lat_deg: float = -87.3,
    faustini_lon_deg: float = 84.5,
    t_start_iso: str = "2024-01-01T00:00:00",
    batch_size: int = 50,
    make_diagnostic_plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Sun's position (azimuth, elevation) as seen from a fixed
    point on the Moon, over a simulated time span.

    Reproduces the orchestration of PSR_Mapping.ipynb cell 21: builds a
    Moon-fixed local frame (up/north/east) at the given lunar coordinates,
    computes the Sun's barycentric position at each timestep via
    Astropy's DE421 ephemeris, rotates it from ICRS into the
    Moon-Centred Moon-Fixed (MCMF) frame using
    ``build_icrs_to_mcmf_rotation``, and projects it onto the local frame
    to get azimuth (0 deg = North, clockwise) and elevation.

    If ``sun_positions.npy`` and ``sun_times.npy`` already exist in
    ``output_dir`` with the expected shape, they are loaded instead of
    recomputed (resume behavior identical to the original cell).

    Parameters
    ----------
    output_dir : str
        Directory to load/save ``sun_positions.npy`` and
        ``sun_times.npy``, and the diagnostic plot, from/to.
    sim_years : float, default 1
        Simulation duration in years.
    sun_timestep_h : float, default 6
        Hours between Sun-position samples.
    faustini_lat_deg, faustini_lon_deg : float, default -87.3, 84.5
        Lunar planetographic latitude/longitude of the observation point
        (Faustini crater).
    t_start_iso : str, default "2024-01-01T00:00:00"
        Start time of the simulation (TDB scale).
    batch_size : int, default 50
        Number of timesteps processed per ephemeris batch call.
    make_diagnostic_plot : bool, default True
        Whether to generate and save the diagnostic plot panel.

    Returns
    -------
    tuple of np.ndarray
        ``(sun_positions, sun_times_unix)`` where ``sun_positions`` has
        shape ``(n_steps, 2)`` with columns ``[azimuth_rad, elevation_rad]``,
        and ``sun_times_unix`` has shape ``(n_steps,)``.
    """
    ensure_dir(output_dir)
    sun_pos_path = os.path.join(output_dir, "sun_positions.npy")
    sun_time_path = os.path.join(output_dir, "sun_times.npy")
    diag_path = os.path.join(output_dir, "07_solar_ephemeris.png")

    # Check whether MCMF is available in this Astropy build (informational
    # only — a manual IAU rotation, build_icrs_to_mcmf_rotation, is used
    # regardless, matching the original notebook's actual behavior).
    try:
        from astropy.coordinates import MCMF  # noqa: F401
        print("MCMF frame available")
    except ImportError:
        print("MCMF not available in this Astropy build "
              "— using manual IAU rotation (as the notebook does regardless)")

    # ── Build time array ───────────────────────────────────────────
    t_start = Time(t_start_iso, format="isot", scale="tdb")
    n_steps = int(sim_years * 365.25 * 24 / sun_timestep_h)
    dt_hours = np.arange(n_steps, dtype=np.float64) * sun_timestep_h
    times = t_start + dt_hours * u.hour

    print(f"Simulation span    : {sim_years} year(s) = {n_steps} steps")
    print(f"Time step          : {sun_timestep_h} hours")
    print(f"Faustini lat/lon   : {faustini_lat_deg} deg / {faustini_lon_deg} degE")

    # ── Build Moon-fixed local frame at the observation point ──────
    lat_r = np.radians(faustini_lat_deg)
    lon_r = np.radians(faustini_lon_deg)

    obs_mcmf = R_MOON_M * np.array([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r)
    ])
    up_mcmf = obs_mcmf / np.linalg.norm(obs_mcmf)

    north_mcmf = np.array([
        -np.sin(lat_r) * np.cos(lon_r),
        -np.sin(lat_r) * np.sin(lon_r),
        np.cos(lat_r)
    ])
    north_mcmf /= np.linalg.norm(north_mcmf)

    east_mcmf = np.cross(north_mcmf, up_mcmf)
    east_mcmf /= np.linalg.norm(east_mcmf)

    # ── Resume check ────────────────────────────────────────────────
    skip_compute = False
    if os.path.exists(sun_pos_path) and os.path.exists(sun_time_path):
        existing = np.load(sun_pos_path)
        if existing.shape == (n_steps, 2):
            print(f"sun_positions.npy exists with correct shape "
                  f"{existing.shape} — loading.")
            sun_positions = existing
            sun_times_unix = np.load(sun_time_path)
            skip_compute = True
        else:
            print(f"Shape mismatch ({existing.shape} vs ({n_steps},2)) "
                  f"— recomputing.")

    # ── Main computation ─────────────────────────────────────────────
    if not skip_compute:
        solar_system_ephemeris.set("builtin")

        sun_az_arr = np.zeros(n_steps, dtype=np.float32)
        sun_elev_arr = np.zeros(n_steps, dtype=np.float32)

        t_start_comp = time.time()
        print(f"\nComputing {n_steps} Sun positions (DE421 + IAU rotation)...")

        for b0 in range(0, n_steps, batch_size):
            b1 = min(b0 + batch_size, n_steps)
            batch_times = times[b0:b1]

            moon_bary, _ = get_body_barycentric_posvel("moon", batch_times)
            sun_bary, _ = get_body_barycentric_posvel("sun", batch_times)

            sun_rel_icrs = (
                (sun_bary.xyz - moon_bary.xyz).to(u.m).value
            )

            for i in range(b1 - b0):
                R = build_icrs_to_mcmf_rotation(batch_times[i])
                sun_mcmf = R @ sun_rel_icrs[:, i]

                s = sun_mcmf / np.linalg.norm(sun_mcmf)
                s_up = np.dot(s, up_mcmf)
                s_north = np.dot(s, north_mcmf)
                s_east = np.dot(s, east_mcmf)

                horiz = np.sqrt(s_north ** 2 + s_east ** 2)
                elev = np.arctan2(s_up, horiz)  # radians
                az = np.arctan2(s_east, s_north) % (2 * np.pi)

                sun_az_arr[b0 + i] = np.float32(az)
                sun_elev_arr[b0 + i] = np.float32(elev)

            if b0 % 200 == 0 or b1 == n_steps:
                elapsed = time.time() - t_start_comp
                frac = b1 / n_steps
                eta = (elapsed / frac - elapsed) if frac > 0.01 else 0
                print(f"  {b1:4d}/{n_steps} steps | "
                      f"{elapsed:5.1f}s elapsed | ETA {eta:5.0f}s")

        sun_positions = np.stack([sun_az_arr, sun_elev_arr], axis=1)
        sun_times_unix = np.array([t.unix for t in times], dtype=np.float64)

        np.save(sun_pos_path, sun_positions)
        np.save(sun_time_path, sun_times_unix)

        total_t = time.time() - t_start_comp
        print(f"Saved: {sun_pos_path}")
        print(f"Saved: {sun_time_path}")
        print(f"Total compute time: {total_t:.1f}s ({total_t / 60:.1f} min)")

    # ── Validation ────────────────────────────────────────────────
    az_deg = np.degrees(sun_positions[:, 0])
    elev_deg = np.degrees(sun_positions[:, 1])

    checks = {
        "N steps matches config": len(elev_deg) == n_steps,
        "No NaN in elevation": not np.any(np.isnan(elev_deg)),
        "No NaN in azimuth": not np.any(np.isnan(az_deg)),
        "Max elevation <= +2.0 deg": np.max(elev_deg) <= 2.0,
        "Min elevation >= -2.0 deg": np.min(elev_deg) >= -2.0,
        "Azimuth covers full 360 deg": (az_deg.max() - az_deg.min()) > 300,
        "Some steps above horizon": np.sum(elev_deg > 0) > 100,
        "Some steps below horizon": np.sum(elev_deg < 0) > 100,
    }
    for label, result in checks.items():
        print(f"  {'OK' if result else 'FAIL'}  {label}")

    pct_above = np.sum(elev_deg > 0) / n_steps * 100
    print(f"Sun elevation range : {np.min(elev_deg):.4f} to {np.max(elev_deg):.4f} deg")
    print(f"Above horizon       : {pct_above:.1f}% of year")

    # ── Diagnostic plots ─────────────────────────────────────────────
    if make_diagnostic_plot:
        day_axis = dt_hours / 24.0
        fig = plt.figure(figsize=(18, 11), facecolor='#0a0a0a')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

        def sa(ax):
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for sp in ax.spines.values():
                sp.set_edgecolor('#444')
            return ax

        ax = sa(fig.add_subplot(gs[0, 0]))
        ax.plot(day_axis, elev_deg, color='#EF9F27', lw=0.8)
        ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.5, label='horizon')
        ax.axhline(1.54, color='#FF4444', lw=0.8, ls=':', label='+/-1.54 deg (Moon tilt)')
        ax.axhline(-1.54, color='#FF4444', lw=0.8, ls=':')
        ax.set_xlabel('Day of simulation')
        ax.set_ylabel('Sun elevation (deg)')
        ax.set_title('Sun Elevation at Faustini', color='white')
        ax.legend(facecolor='#111', labelcolor='white', fontsize=8)

        ax = sa(fig.add_subplot(gs[0, 1]))
        ax.plot(day_axis, az_deg, color='#2ECC40', lw=0.8)
        ax.set_xlabel('Day of simulation')
        ax.set_ylabel('Sun azimuth (deg)')
        ax.set_title('Sun Azimuth (full 0-360 deg expected)', color='white')

        ax = sa(fig.add_subplot(gs[0, 2]))
        ax.hist(elev_deg, bins=60, color='#3B8BD4', edgecolor='none')
        ax.axvline(0, color='white', lw=1, ls='--', label='horizon')
        ax.set_xlabel('Elevation (deg)')
        ax.set_ylabel('Count')
        ax.set_title('Elevation Distribution', color='white')
        ax.legend(facecolor='#111', labelcolor='white', fontsize=8)

        ax_p = fig.add_subplot(gs[1, 0], projection='polar', facecolor='#111')
        ax_p.set_theta_zero_location('N')
        ax_p.set_theta_direction(-1)
        zen_dist = 90.0 - elev_deg
        sc = ax_p.scatter(np.radians(az_deg), zen_dist,
                          c=elev_deg, cmap='RdYlGn',
                          s=1.5, alpha=0.8, vmin=-1.54, vmax=1.54)
        ax_p.set_title('Polar Sun Track\nGreen = above horizon  Red = below',
                       color='white', pad=15)
        ax_p.tick_params(colors='white')
        plt.colorbar(sc, ax=ax_p, pad=0.1).set_label('Elevation (deg)', color='white')

        ax = sa(fig.add_subplot(gs[1, 1]))
        pct_below = 100.0 - pct_above
        bars = ax.bar(['Above\nhorizon', 'Below\nhorizon'],
                      [pct_above, pct_below],
                      color=['#2ECC40', '#FF4136'], width=0.5)
        ax.set_ylabel('% of year')
        ax.set_title(f'Illumination Fraction\n{pct_above:.1f}% above horizon',
                     color='white')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}%',
                    ha='center', color='white', fontsize=10)

        ax = sa(fig.add_subplot(gs[1, 2]))
        sc2 = ax.scatter(az_deg, elev_deg, c=day_axis, cmap='plasma', s=1.5, alpha=0.7)
        ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.6)
        ax.set_xlabel('Azimuth (deg)')
        ax.set_ylabel('Elevation (deg)')
        ax.set_title('Az vs Elev (colour = day)', color='white')
        plt.colorbar(sc2, ax=ax, pad=0.02).set_label('Day', color='white')

        plt.suptitle(
            'Solar Ephemeris | Faustini, Lunar South Pole\n'
            'DE421 + IAU WGCCRE Moon Rotation | Mazarico et al. 2011',
            color='white', fontsize=12, y=1.01)

        plt.savefig(diag_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        plt.show()
        print(f"Diagnostic saved: {diag_path}")

    return sun_positions, sun_times_unix


def compute_illumination_and_psr(
    horizon_path: str,
    sun_positions_path: str,
    output_dir: str,
    dem_path: Optional[str] = None,
    chunk_rows: int = 100,
    make_diagnostic_plots: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the annual illumination fraction and derive the boolean PSR
    mask for every DEM pixel.

    Reproduces the orchestration of PSR_Mapping.ipynb cell 23. Core
    algorithm (Mazarico et al. 2011, eq. 3): for pixel ``(r, c)`` at
    timestep ``t`` with Sun azimuth index ``k``,
    ``illuminated = sun_elevation[t] > horizon_angles[k, r, c]``;
    ``illumination_fraction(r, c) = sum(illuminated) / n_times``.
    A pixel is classified as PSR if its illumination fraction is exactly
    0.0 across the full simulated period.

    Processing is done in row-chunks (resumable, matching the original
    cell) to keep memory use bounded.

    Parameters
    ----------
    horizon_path : str
        Path to ``horizon_angles.npy`` (from ``horizon_angles.compute_horizon_angles``).
    sun_positions_path : str
        Path to ``sun_positions.npy`` (from ``compute_solar_ephemeris``).
    output_dir : str
        Directory to save ``illumination_fraction.npy``, ``psr_mask.npy``,
        chunk files, and diagnostic plots into.
    dem_path : str, optional
        Path to the DEM GeoTIFF, used only for the DEM-context panel in
        the diagnostic plots. If not provided, plotting is skipped even
        if ``make_diagnostic_plots=True``.
    chunk_rows : int, default 100
        Number of rows processed per resumable chunk.
    make_diagnostic_plots : bool, default True
        Whether to generate and save the illumination/PSR diagnostic
        figures (requires ``dem_path``).

    Returns
    -------
    tuple of np.ndarray
        ``(illumination_fraction, psr_mask)``, both shape ``(rows, cols)``;
        ``psr_mask`` is boolean.
    """
    illum_chunk_dir = os.path.join(output_dir, "illum_chunks")
    illum_path = os.path.join(output_dir, "illumination_fraction.npy")
    psr_path = os.path.join(output_dir, "psr_mask.npy")
    illum_png = os.path.join(output_dir, "08_illumination_fraction.png")
    psr_png = os.path.join(output_dir, "08_psr_mask.png")
    ensure_dir(illum_chunk_dir)

    print("Loading inputs...")
    horizon_angles = np.load(horizon_path)  # (n_az, rows, cols) radians
    sun_positions = np.load(sun_positions_path)  # (n_times, 2)

    n_az, n_rows, n_cols = horizon_angles.shape
    n_times = sun_positions.shape[0]
    sun_az_all = sun_positions[:, 0]
    sun_el_all = sun_positions[:, 1]

    dem = None
    if dem_path is not None:
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan

    print(f"  horizon_angles : {horizon_angles.shape} dtype={horizon_angles.dtype}")
    print(f"  sun_positions  : {sun_positions.shape}")
    print(f"  n_azimuths     : {n_az} ({360 // n_az} deg spacing)")

    # ── Verify azimuth convention consistency ──────────────────────
    # horizon_angles azimuth index k -> angle = k * (360/n_az) degrees.
    # sun_positions azimuth: 0 = North, clockwise, range 0-2*pi.
    # Both use the identical convention, so direct index mapping is valid.
    az_step_rad = 2.0 * np.pi / n_az
    az_indices = (np.round(sun_az_all / az_step_rad).astype(np.int32) % n_az)

    print(f"Azimuth bin mapping: step={np.degrees(az_step_rad):.1f} deg, "
          f"index range {az_indices.min()}-{az_indices.max()} "
          f"(expected 0-{n_az - 1})")

    # ── Illumination computation - chunked + resume ────────────────
    n_chunks = (n_rows + chunk_rows - 1) // chunk_rows

    def _chunk_path(ch: int) -> str:
        return chunk_path(illum_chunk_dir, "illum_chunk_", ch)

    done_chunks = [os.path.exists(_chunk_path(i)) for i in range(n_chunks)]
    print(f"Chunks total: {n_chunks} ({chunk_rows} rows each), "
          f"already done: {sum(done_chunks)}")

    t_total = time.time()
    for ch in range(n_chunks):
        if done_chunks[ch]:
            print(f"  Chunk {ch + 1:02d}/{n_chunks} — already done, skipping")
            continue

        r0 = ch * chunk_rows
        r1 = min(r0 + chunk_rows, n_rows)
        hz_chunk = horizon_angles[:, r0:r1, :]
        chunk_h = r1 - r0

        lit_count = np.zeros((chunk_h, n_cols), dtype=np.int32)
        t_chunk = time.time()

        for t_idx in range(n_times):
            k = az_indices[t_idx]
            sun_elev_t = sun_el_all[t_idx]
            hz_at_k = hz_chunk[k, :, :]
            lit_count += (sun_elev_t > hz_at_k).astype(np.int32)

        np.save(_chunk_path(ch), lit_count)

        elapsed = time.time() - t_chunk
        done_now = sum(os.path.exists(_chunk_path(i)) for i in range(n_chunks))
        remaining = n_chunks - done_now
        eta = elapsed * remaining
        print(f"  Chunk {ch + 1:02d}/{n_chunks} rows {r0:4d}-{r1:4d} | "
              f"{elapsed:.1f}s | ETA {eta / 60:.1f} min")

    print(f"All chunks complete. Total wall time: {(time.time() - t_total) / 60:.1f} min")

    # ── Merge chunks -> illumination fraction ───────────────────────
    lit_count_full = np.zeros((n_rows, n_cols), dtype=np.int32)
    for ch in range(n_chunks):
        r0 = ch * chunk_rows
        r1 = min(r0 + chunk_rows, n_rows)
        lit_count_full[r0:r1, :] = np.load(_chunk_path(ch))

    illum_frac = lit_count_full.astype(np.float32) / n_times

    # ── PSR mask ─────────────────────────────────────────────────────
    # PSR definition (Mazarico 2011): illumination_fraction == 0.0 across
    # the full simulated period.
    psr_mask = (illum_frac == 0.0)

    np.save(illum_path, illum_frac)
    np.save(psr_path, psr_mask)
    print(f"Saved: {illum_path}")
    print(f"Saved: {psr_path}")

    # ── Diagnostic statistics ──────────────────────────────────────
    total_px = n_rows * n_cols
    psr_px = int(np.sum(psr_mask))
    illum_px = total_px - psr_px
    psr_pct = psr_px / total_px * 100
    illum_pct = illum_px / total_px * 100
    psr_area_km = psr_px * (100 * 100) / 1e6  # 100 m/pixel -> km^2

    illum_nonzero = illum_frac[~psr_mask]

    print(f"Total pixels: {total_px:,}  Illuminated: {illum_px:,} ({illum_pct:.1f}%)  "
          f"PSR: {psr_px:,} ({psr_pct:.1f}%)  PSR area: {psr_area_km:.1f} km^2")
    print(f"Mean illum (non-PSR): {illum_nonzero.mean() * 100:.1f}%  "
          f"Max illum fraction: {illum_frac.max() * 100:.1f}%")
    print(f"Faustini PSR reference (Mazarico 2011): ~1180 km^2 | "
          f"Your result: {psr_area_km:.0f} km^2")
    if abs(psr_area_km - 1180) / 1180 < 0.30:
        print("Within 30% of published value — scientifically plausible")
    else:
        print("Outside 30% of published value — check DEM coverage / N_AZIMUTHS")

    # ── Diagnostic plots ─────────────────────────────────────────────
    if make_diagnostic_plots and dem is not None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='#0a0a0a')
        for ax in axes:
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for sp in ax.spines.values():
                sp.set_edgecolor('#555')

        im0 = axes[0].imshow(dem, cmap='gray', origin='upper')
        cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cb0.set_label('Elevation (m)', color='white')
        plt.setp(cb0.ax.yaxis.get_ticklabels(), color='white')
        axes[0].set_title('DEM — Faustini region\n(context)', color='white', fontsize=11)
        axes[0].set_xlabel('Column (pixels)')
        axes[0].set_ylabel('Row (pixels)')

        cmap_illum = plt.cm.get_cmap('inferno').copy()
        cmap_illum.set_under('#000000')
        im1 = axes[1].imshow(illum_frac * 100, cmap=cmap_illum,
                             vmin=0.01, vmax=100, origin='upper')
        cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cb1.set_label('Illumination fraction (%)', color='white')
        plt.setp(cb1.ax.yaxis.get_ticklabels(), color='white')
        axes[1].set_title('Annual Illumination Fraction\nBlack = PSR (0%)',
                          color='white', fontsize=11)
        axes[1].set_xlabel('Column (pixels)')
        axes[1].set_ylabel('Row (pixels)')

        axes[2].hist(illum_nonzero * 100, bins=80, color='#EF9F27',
                    edgecolor='none', alpha=0.85)
        axes[2].set_xlabel('Illumination fraction (%)')
        axes[2].set_ylabel('Pixel count')
        axes[2].set_title('Illumination Distribution\n(PSR pixels excluded)',
                          color='white', fontsize=11)
        axes[2].xaxis.set_major_formatter(PercentFormatter())

        plt.suptitle(
            'Annual Illumination Fraction | Faustini, Lunar South Pole\n'
            f'Mazarico et al. (2011) method | {n_times} timesteps x '
            f'6 h | {n_az} azimuths',
            color='white', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(illum_png, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        plt.show()
        print(f"Saved: {illum_png}")

        fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='#0a0a0a')
        for ax in axes:
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for sp in ax.spines.values():
                sp.set_edgecolor('#555')

        axes[0].imshow(dem, cmap='gray', origin='upper', alpha=0.75)
        psr_overlay = np.where(psr_mask, 1.0, np.nan)
        axes[0].imshow(psr_overlay, cmap=mcolors.ListedColormap(['#3B8BD4']),
                      origin='upper', alpha=0.75, vmin=0, vmax=1)
        axes[0].set_title('PSR overlaid on DEM\nBlue = permanently shadowed',
                          color='white', fontsize=11)
        axes[0].set_xlabel('Column (pixels)')
        axes[0].set_ylabel('Row (pixels)')

        axes[1].imshow(psr_mask.astype(np.float32),
                      cmap=mcolors.ListedColormap(['#1a1a2e', '#3B8BD4']),
                      origin='upper', vmin=0, vmax=1)
        axes[1].set_title(f'PSR Mask\n{psr_px:,} pixels | '
                          f'{psr_area_km:.0f} km2 | {psr_pct:.1f}% of domain',
                          color='white', fontsize=11)
        axes[1].set_xlabel('Column (pixels)')
        axes[1].set_ylabel('Row (pixels)')

        im2 = axes[2].imshow(illum_frac * 100, cmap='hot', vmin=0, vmax=100,
                             origin='upper')
        axes[2].contour(psr_mask.astype(float), levels=[0.5],
                       colors=['#00FFFF'], linewidths=[0.8])
        cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        cb2.set_label('Illumination (%)', color='white')
        plt.setp(cb2.ax.yaxis.get_ticklabels(), color='white')
        axes[2].set_title('Illumination + PSR boundary\nCyan contour = PSR edge',
                          color='white', fontsize=11)
        axes[2].set_xlabel('Column (pixels)')
        axes[2].set_ylabel('Row (pixels)')

        plt.suptitle(
            f'PSR Mask | Faustini, Lunar South Pole\n'
            f'PSR area: {psr_area_km:.0f} km2 | '
            f'Reference (Mazarico 2011): ~1180 km2',
            color='white', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(psr_png, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        plt.show()
        print(f"Saved: {psr_png}")

    return illum_frac, psr_mask


def visualize_psr_mask(psr_mask_path: str) -> float:
    """
    Display the PSR mask and report the fraction of the domain it covers.

    Reproduced from PSR_Mapping.ipynb cell 26.

    Parameters
    ----------
    psr_mask_path : str
        Path to ``psr_mask.npy``.

    Returns
    -------
    float
        The PSR pixel fraction (``psr_mask.mean()``).
    """
    psr = np.load(psr_mask_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(psr, cmap='gray')
    plt.title("PSR Mask")
    plt.colorbar()
    plt.show()

    fraction = psr.mean()
    print("PSR fraction =", fraction)
    return fraction


def compute_regional_psr_area(
    psr_mask_path: str,
    center_col: int = 980,
    center_row: int = 190,
    radius_px: int = 210,
    pixel_area_km2: float = 0.01,
) -> float:
    """
    Compute the PSR area within a circular region of interest (e.g. the
    Faustini crater floor) rather than the full DEM extent.

    Reproduced from PSR_Mapping.ipynb cell 27. The default center/radius
    values are the specific pixel-space crop used in the notebook to
    isolate Faustini crater within this DEM's grid; they are specific to
    this DEM's extent and origin, not a general-purpose default.

    Parameters
    ----------
    psr_mask_path : str
        Path to ``psr_mask.npy``.
    center_col, center_row : int, default 980, 190
        Pixel-space center of the circular region of interest.
    radius_px : int, default 210
        Radius of the region of interest, in pixels.
    pixel_area_km2 : float, default 0.01
        Area of one pixel in km^2 (0.01 km^2 = 100m x 100m pixel).

    Returns
    -------
    float
        PSR area within the region of interest, in km^2.
    """
    psr = np.load(psr_mask_path)
    rows, cols = psr.shape

    Y, X = np.ogrid[:rows, :cols]
    region = ((X - center_col) ** 2 + (Y - center_row) ** 2) <= radius_px ** 2

    psr_region = psr & region
    area_km2 = psr_region.sum() * pixel_area_km2

    print(f"Faustini-only PSR area = {area_km2} km2")
    return area_km2