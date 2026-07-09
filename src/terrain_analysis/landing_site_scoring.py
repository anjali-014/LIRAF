"""
Final landing-site selection and ranking, Faustini crater.

Extracted from PSR_Mapping.ipynb, "CELL 10 — Final Landing Site
Selection". Scientific basis:
  - Hazard avoidance : Arvidson et al. (2002); slope < 10 deg
  - Illumination      : Mazarico et al. (2011); > 20% for power/comms
  - Ice proximity     : Rubanenko et al. (2019); traverse < 5 km
  - Roughness proxy   : slope std-dev window (Kreslavsky & Head, 2000)
  - Scoring model     : weighted additive (ESA/ISRO site-selection practice)

No weight, threshold, or formula has been changed from the original
notebook cell. Surrounding I/O/plotting/save logic has been wrapped into
a single callable function with explicit parameters in place of notebook
globals and hardcoded Drive paths.
"""

import os
import json
import warnings
from typing import List, Dict, Optional

import numpy as np
import rasterio
from rasterio.transform import xy as rio_xy
from scipy.ndimage import (
    uniform_filter, label, distance_transform_edt, sobel,
)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from .utils import ensure_dir

warnings.filterwarnings("ignore")

# Pixel geometry
PIXEL_M = 100
PIXEL_KM2 = (PIXEL_M / 1000) ** 2

# Scoring weights (must sum to 1.0) — reproduced from cell 34
W_SLOPE = 0.35
W_ILLUM = 0.25
W_ICE_PROX = 0.25
W_ROUGHNESS = 0.15

MIN_SITE_AREA_KM2 = 0.1


def norm01(arr: np.ndarray, lo_pct: float = 1, hi_pct: float = 99) -> np.ndarray:
    """
    Robust normalization to [0, 1] using percentile clipping.

    Reproduced verbatim from PSR_Mapping.ipynb cell 34.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    lo_pct, hi_pct : float, default 1, 99
        Lower/upper percentiles used as the normalization bounds.

    Returns
    -------
    np.ndarray
        Array clipped to [0, 1], same shape as input, dtype float32.
    """
    lo = np.nanpercentile(arr, lo_pct)
    hi = np.nanpercentile(arr, hi_pct)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0).astype(np.float32)


def style_ax(ax, title: str, subtitle: str = "") -> None:
    """
    Apply consistent dark-theme title/subtitle styling to a Matplotlib
    axis.

    Reproduced verbatim from PSR_Mapping.ipynb cell 34. Note: a
    differently-implemented axis-styling helper of the same conceptual
    purpose also exists in ``ice_trap_detection.label_ax`` (from cell
    30) — kept separate rather than merged, since the two were defined
    independently in the original notebook with different styling
    details (this one also sets ``ax.set_facecolor``, which
    ``label_ax`` does not).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style.
    title : str
        Main title text.
    subtitle : str, default ""
        Optional subtitle text.
    """
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=5)
    if subtitle:
        ax.text(0.5, 1.005, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", color="#888888", fontsize=7.5)


def rank_landing_sites(
    dem_path: str,
    psr_mask_path: str,
    illumination_path: str,
    ice_trap_path: str,
    output_dir: str,
    make_diagnostic_plot: bool = True,
) -> Dict[str, object]:
    """
    Rank candidate landing sites using a weighted composite of slope
    safety, illumination, ice-trap proximity, and terrain roughness, and
    identify the primary recommended site.

    Reproduces the full orchestration of PSR_Mapping.ipynb cell 34:

    1. Load DEM, PSR mask, illumination fraction, candidate ice-trap
       stack (composite score / candidate mask / high-confidence mask).
    2. Compute terrain safety layers: slope (Sobel-based, degrees),
       terrain roughness (local slope std-dev in a 500 m window, per
       Kreslavsky & Head 2000), and Euclidean distance (km) to the
       nearest high-confidence ice trap.
    3. Apply hard exclusion criteria: slope < 10 deg (Arvidson 2002),
       illumination > 20% (Mazarico 2011), roughness < 5 deg, and
       excluding deep PSR interiors (illumination < 5%).
    4. Compute composite landing score as a weighted sum of normalized
       slope/illumination/ice-proximity/roughness scores (weights:
       slope 0.35, illum 0.25, ice-proximity 0.25, roughness 0.15).
    5. Threshold at the 70th percentile of landable scores, connected-
       component label (8-connectivity), drop clusters below 0.1 km^2,
       compute per-site statistics and nearest-ice-trap distance, rank
       by mean score (tie-broken by proximity to ice).
    6. Save the 3-layer array, full site catalogue, the best-site
       recommendation (with a scientific justification string built
       from that site's own statistics), a resume checkpoint, and
       (optionally) a 6-panel publication figure.

    No weight, threshold, or formula differs from cell 34.

    Parameters
    ----------
    dem_path : str
        Path to the DEM GeoTIFF.
    psr_mask_path : str
        Path to ``psr_mask.npy``.
    illumination_path : str
        Path to ``illumination_fraction.npy``.
    ice_trap_path : str
        Path to ``candidate_ice_traps.npy`` (3-layer stack from
        ``ice_trap_detection.detect_candidate_ice_traps``).
    output_dir : str
        Directory to save ``landing_sites.npy``,
        ``landing_site_catalogue.json``, ``best_landing_site.json``,
        ``cell10_checkpoint.npz``, and the diagnostic figure into.
    make_diagnostic_plot : bool, default True
        Whether to generate and save the 6-panel diagnostic figure.

    Returns
    -------
    dict
        ``{"landing_score": np.ndarray, "site_pixels": np.ndarray (bool),
        "landable": np.ndarray (bool), "sites": list[dict],
        "best_site": dict}``.
    """
    ensure_dir(output_dir)
    out_npy = os.path.join(output_dir, "landing_sites.npy")
    out_cat = os.path.join(output_dir, "landing_site_catalogue.json")
    out_best = os.path.join(output_dir, "best_landing_site.json")
    out_png = os.path.join(output_dir, "10_landing_site_selection.png")
    checkpoint_path = os.path.join(output_dir, "cell10_checkpoint.npz")

    print("=" * 64)
    print("Final Landing Site Selection — Faustini Crater")
    print("=" * 64)

    # ── STEP 1: Load all inputs ─────────────────────────────────
    print("\n[1/8] Loading inputs...")

    psr_mask = np.load(psr_mask_path).astype(bool)
    illum = np.load(illumination_path).astype(np.float32)
    ice_stack = np.load(ice_trap_path)
    candidate_mask = ice_stack[1].astype(bool)
    high_conf_mask = ice_stack[2].astype(bool)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if nodata is not None:
        dem[dem == nodata] = np.nan

    rows, cols = dem.shape
    print(f"    DEM            : {rows} x {cols}  "
          f"({rows * PIXEL_M / 1000:.1f} x {cols * PIXEL_M / 1000:.1f} km)")
    print(f"    PSR pixels     : {psr_mask.sum():,}  ({psr_mask.mean() * 100:.1f}%)")
    print(f"    High-conf ice  : {high_conf_mask.sum():,} px  "
          f"({high_conf_mask.sum() * PIXEL_KM2:.2f} km^2)")

    # ── STEP 2: Terrain safety layers ───────────────────────────
    print("\n[2/8] Computing terrain safety layers...")

    dem_filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dz_dx = sobel(dem_filled, axis=1) / (8.0 * PIXEL_M)
    dz_dy = sobel(dem_filled, axis=0) / (8.0 * PIXEL_M)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))).astype(np.float32)
    slope_deg[np.isnan(dem)] = np.nan

    window = 5
    slope_sq_mean = uniform_filter(slope_deg ** 2, size=window)
    slope_mean_sq = uniform_filter(slope_deg, size=window) ** 2
    roughness = np.sqrt(np.maximum(slope_sq_mean - slope_mean_sq, 0)).astype(np.float32)
    roughness[np.isnan(dem)] = np.nan

    dist_to_ice_px = distance_transform_edt(~high_conf_mask).astype(np.float32)
    dist_to_ice_km = dist_to_ice_px * (PIXEL_M / 1000)

    print(f"    Slope range        : {np.nanmin(slope_deg):.1f} - {np.nanmax(slope_deg):.1f} deg")
    print(f"    Roughness range    : {np.nanmin(roughness):.2f} - {np.nanmax(roughness):.2f} deg")
    print(f"    Max dist-to-ice    : {np.nanmax(dist_to_ice_km):.1f} km")

    # ── STEP 3: Hard exclusion masks ────────────────────────────
    print("\n[3/8] Applying hard exclusion criteria...")

    safe_slope = slope_deg < 10.0
    safe_illum = illum > 0.20
    safe_roughness = roughness < 5.0
    not_psr_deep = ~(psr_mask & (illum < 0.05))

    landable = (safe_slope & safe_illum & safe_roughness &
                not_psr_deep & ~np.isnan(dem))

    n_landable = landable.sum()
    print(f"    Safe slope   (<10 deg)  : {safe_slope.sum():,} px  ({safe_slope.mean() * 100:.1f}%)")
    print(f"    Safe illum   (>20%)     : {safe_illum.sum():,} px  ({safe_illum.mean() * 100:.1f}%)")
    print(f"    Safe rough   (<5 deg)   : {safe_roughness.sum():,} px  ({safe_roughness.mean() * 100:.1f}%)")
    print(f"    Combined landable       : {n_landable:,} px  ({n_landable * PIXEL_KM2:.2f} km^2)")

    if n_landable == 0:
        raise RuntimeError(
            "No landable pixels found. Check DEM coverage or loosen thresholds."
        )

    # ── STEP 4: Composite landing score ─────────────────────────
    print("\n[4/8] Computing composite landing score...")

    slope_score = norm01(30.0 - slope_deg)
    illum_score = norm01(illum)
    prox_score = norm01(50.0 - dist_to_ice_km)
    rough_score = norm01(10.0 - roughness)

    landing_score = (
        W_SLOPE * slope_score +
        W_ILLUM * illum_score +
        W_ICE_PROX * prox_score +
        W_ROUGHNESS * rough_score
    ).astype(np.float32)

    landing_score[~landable] = 0.0
    landing_score[np.isnan(dem)] = 0.0

    print(f"    Score range (landable): "
          f"{landing_score[landable].min():.4f} - {landing_score[landable].max():.4f}")
    print(f"    Mean landing score    : {landing_score[landable].mean():.4f}")

    # ── STEP 5: Connected-component site identification ─────────
    print("\n[5/8] Identifying and ranking landing site clusters...")

    threshold = np.percentile(landing_score[landable], 70)
    site_pixels = (landing_score >= threshold) & landable

    structure = np.ones((3, 3), dtype=int)
    labeled, n_sites = label(site_pixels, structure=structure)
    print(f"    Score threshold (top 30%): {threshold:.4f}")
    print(f"    Candidate site clusters  : {n_sites}")

    sites: List[Dict] = []
    t = None
    for sid in range(1, n_sites + 1):
        mask_s = labeled == sid
        area_km2 = mask_s.sum() * PIXEL_KM2
        if area_km2 < MIN_SITE_AREA_KM2:
            continue

        sc = landing_score[mask_s]
        sl = slope_deg[mask_s]
        il = illum[mask_s]
        di = dist_to_ice_km[mask_s]
        ro = roughness[mask_s]
        el = dem[mask_s]

        rr, cc = np.where(mask_s)
        w = sc / sc.sum()
        cr = (rr * w).sum()
        cc_w = (cc * w).sum()

        cx, cy = rio_xy(transform, cr, cc_w)
        if crs and crs.is_geographic:
            lon, lat = float(cx), float(cy)
        else:
            try:
                from pyproj import Transformer
                t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                lon, lat = t.transform(float(cx), float(cy))
            except Exception:
                lon, lat = float(cx), float(cy)

        ice_rows, ice_cols = np.where(high_conf_mask)
        if len(ice_rows):
            dists = np.sqrt((ice_rows - cr) ** 2 + (ice_cols - cc_w) ** 2)
            nearest_idx = dists.argmin()
            ice_r = int(ice_rows[nearest_idx])
            ice_c = int(ice_cols[nearest_idx])
            ice_cx, ice_cy = rio_xy(transform, ice_r, ice_c)
            if crs and crs.is_geographic:
                ice_lon, ice_lat = float(ice_cx), float(ice_cy)
            else:
                try:
                    ice_lon, ice_lat = t.transform(float(ice_cx), float(ice_cy))
                except Exception:
                    ice_lon, ice_lat = float(ice_cx), float(ice_cy)
        else:
            ice_r = ice_c = ice_lat = ice_lon = None

        sites.append({
            "rank": 0,
            "site_id": int(sid),
            "area_km2": round(float(area_km2), 3),
            "mean_score": round(float(sc.mean()), 4),
            "max_score": round(float(sc.max()), 4),
            "mean_slope_deg": round(float(np.nanmean(sl)), 2),
            "max_slope_deg": round(float(np.nanmax(sl)), 2),
            "mean_illumination": round(float(il.mean()), 4),
            "mean_dist_ice_km": round(float(di.mean()), 3),
            "min_dist_ice_km": round(float(di.min()), 3),
            "mean_roughness_deg": round(float(np.nanmean(ro)), 3),
            "mean_elevation_m": round(float(np.nanmean(el)), 1),
            "centroid_lat": round(lat, 5),
            "centroid_lon": round(lon, 5),
            "pixel_row": int(cr),
            "pixel_col": int(cc_w),
            "nearest_ice_row": ice_r,
            "nearest_ice_col": ice_c,
            "nearest_ice_lat": round(ice_lat, 5) if ice_lat else None,
            "nearest_ice_lon": round(ice_lon, 5) if ice_lon else None,
        })

    sites.sort(key=lambda x: (-x["mean_score"], x["min_dist_ice_km"]))
    for i, s in enumerate(sites):
        s["rank"] = i + 1

    print(f"    Sites above {MIN_SITE_AREA_KM2} km^2   : {len(sites)}")

    # ── STEP 6: Save outputs + checkpoint ────────────────────────
    print("\n[6/8] Saving outputs...")

    np.save(out_npy, np.stack([
        landing_score,
        site_pixels.astype(np.float32),
        landable.astype(np.float32),
    ], axis=0))
    print(f"    Saved: {out_npy}")

    with open(out_cat, "w") as f:
        json.dump(sites, f, indent=2)
    print(f"    Saved: {out_cat}")

    best = sites[0] if sites else {}
    if best:
        best["isro_justification"] = (
            f"Site #{best['rank']} is recommended as the primary landing target. "
            f"Mean slope of {best['mean_slope_deg']:.1f} deg is below the 10 deg stability "
            f"threshold (Arvidson 2002), ensuring safe touchdown and rover mobility. "
            f"Mean illumination of {best['mean_illumination'] * 100:.1f}% exceeds the "
            f"20% minimum required for continuous solar power and Earth communications "
            f"(Mazarico 2011). The nearest high-confidence ice-trap region is only "
            f"{best['min_dist_ice_km']:.2f} km away, achievable in a single rover "
            f"traverse session. The site lies adjacent to - but not inside - the deep "
            f"PSR, giving access to both illuminated terrain and the cold-trap ice "
            f"resource, consistent with the dual-objective design of Chandrayaan-3 "
            f"and the ISRO Lunar Polar Exploration mission concept."
        )
    with open(out_best, "w") as f:
        json.dump(best, f, indent=2)
    print(f"    Saved: {out_best}")

    np.savez(checkpoint_path,
             landing_score=landing_score,
             site_pixels=site_pixels,
             landable=landable,
             slope_deg=slope_deg,
             dist_to_ice_km=dist_to_ice_km)
    print(f"    Saved: {checkpoint_path}")

    # ── STEP 7: Publication-quality figure ───────────────────────
    if make_diagnostic_plot:
        print("\n[7/8] Generating publication-quality figure...")

        fig = plt.figure(figsize=(20, 14), facecolor="#080808")
        gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30,
                              left=0.06, right=0.97, top=0.91, bottom=0.06)
        axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

        dem_show = np.where(np.isnan(dem), np.nanmin(dem), dem)

        ax = axes[0]
        ax.imshow(dem_show, cmap="gist_earth", interpolation="nearest", alpha=0.85)

        psr_ov = np.ma.masked_where(~psr_mask, np.ones(dem.shape))
        ice_ov = np.ma.masked_where(~high_conf_mask, np.ones(dem.shape))
        land_ov = np.ma.masked_where(~landable, np.ones(dem.shape))

        ax.imshow(psr_ov, cmap=mcolors.ListedColormap(["#2255cc"]), alpha=0.35, interpolation="nearest")
        ax.imshow(ice_ov, cmap=mcolors.ListedColormap(["#ff2255"]), alpha=0.75, interpolation="nearest")
        ax.imshow(land_ov, cmap=mcolors.ListedColormap(["#44ff88"]), alpha=0.30, interpolation="nearest")

        if best:
            br, bc = best["pixel_row"], best["pixel_col"]
            ax.plot(bc, br, "*", color="#ffdd00", markersize=14, zorder=10,
                    markeredgecolor="white", markeredgewidth=0.8)
            if best["nearest_ice_row"] is not None:
                ax.plot([bc, best["nearest_ice_col"]],
                        [br, best["nearest_ice_row"]],
                        color="#ffdd00", lw=1.4, ls="--", zorder=9, alpha=0.9)

        legend_els = [
            mpatches.Patch(color="#2255cc", alpha=0.7, label="PSR"),
            mpatches.Patch(color="#ff2255", alpha=0.9, label="High-conf ice trap"),
            mpatches.Patch(color="#44ff88", alpha=0.6, label="Safe landing zone"),
            Line2D([0], [0], marker="*", color="#ffdd00", markersize=10,
                  linestyle="none", label="Best landing site"),
            Line2D([0], [0], color="#ffdd00", lw=1.4, ls="--", label="Rover traverse"),
        ]
        ax.legend(handles=legend_els, loc="lower right", fontsize=6.5,
                  facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")
        style_ax(ax, "Overview - Landing Site & Ice Access", "DEM + PSR + ice traps + safe zones")

        ax = axes[1]
        ls_show = np.ma.masked_where(landing_score == 0, landing_score)
        im1 = ax.imshow(ls_show, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        cb1 = plt.colorbar(im1, ax=ax, fraction=0.035, pad=0.02)
        cb1.set_label("Landing score", color="#cccccc", fontsize=8)
        for s in sites[:5]:
            ax.plot(s["pixel_col"], s["pixel_row"], "w^", markersize=6, zorder=5)
            ax.text(s["pixel_col"] + 5, s["pixel_row"], f"#{s['rank']}",
                    color="white", fontsize=6.5, zorder=6)
        style_ax(ax, "Composite Landing Score", "0 = excluded  |  1 = optimal")

        ax = axes[2]
        im2 = ax.imshow(slope_deg, cmap="hot_r", vmin=0, vmax=30, interpolation="nearest")
        cb2 = plt.colorbar(im2, ax=ax, fraction=0.035, pad=0.02)
        cb2.set_label("Slope (deg)", color="#cccccc", fontsize=8)
        safe_s_ov = np.ma.masked_where(~safe_slope, np.ones(dem.shape))
        ax.imshow(safe_s_ov, cmap=mcolors.ListedColormap(["#00ffcc"]),
                  alpha=0.25, interpolation="nearest")
        style_ax(ax, "Slope Map", "cyan tint = slope < 10 deg (safe)")

        ax = axes[3]
        im3 = ax.imshow(illum, cmap="plasma", vmin=0, vmax=1, interpolation="nearest")
        cb3 = plt.colorbar(im3, ax=ax, fraction=0.035, pad=0.02)
        cb3.set_label("Illumination fraction", color="#cccccc", fontsize=8)
        si_ov = np.ma.masked_where(~safe_illum, np.ones(dem.shape))
        ax.imshow(si_ov, cmap=mcolors.ListedColormap(["#ffff00"]),
                  alpha=0.20, interpolation="nearest")
        style_ax(ax, "Annual Illumination Fraction", "yellow tint = > 20% (solar-safe)")

        ax = axes[4]
        dist_show = np.where(landable, dist_to_ice_km, np.nan)
        im4 = ax.imshow(dist_show, cmap="viridis_r",
                        vmin=0, vmax=np.nanpercentile(dist_show, 95),
                        interpolation="nearest")
        cb4 = plt.colorbar(im4, ax=ax, fraction=0.035, pad=0.02)
        cb4.set_label("Distance to ice (km)", color="#cccccc", fontsize=8)
        ax.imshow(ice_ov, cmap=mcolors.ListedColormap(["#ff2255"]),
                  alpha=0.8, interpolation="nearest")
        style_ax(ax, "Distance to Nearest Ice Trap", "within landable pixels only")

        ax = axes[5]
        ax.set_facecolor("#0d0d0d")
        ax.axis("off")
        top10 = sites[:10]
        if top10:
            col_labels = ["#", "Area\nkm2", "Score", "Slope\ndeg", "Illum\n%",
                         "Ice\nkm", "Lat", "Lon"]
            table_data = [[
                str(s["rank"]),
                f"{s['area_km2']:.2f}",
                f"{s['mean_score']:.4f}",
                f"{s['mean_slope_deg']:.1f}",
                f"{s['mean_illumination'] * 100:.1f}",
                f"{s['min_dist_ice_km']:.2f}",
                f"{s['centroid_lat']:.3f}",
                f"{s['centroid_lon']:.3f}",
            ] for s in top10]

            tbl = ax.table(cellText=table_data, colLabels=col_labels,
                          loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7.2)
            tbl.scale(1.0, 1.45)

            for (ri, ci), cell in tbl.get_celld().items():
                if ri == 0:
                    cell.set_facecolor("#1e1e3a")
                elif ri == 1 and sites:
                    cell.set_facecolor("#2a1a0a")
                else:
                    cell.set_facecolor("#111111")
                cell.set_edgecolor("#333333")
                cell.set_text_props(color="#dddddd")

        style_ax(ax, "Top-10 Ranked Landing Sites", "row 1 = recommended primary site")

        fig.suptitle(
            "Faustini Crater - Final Landing Site Selection\n"
            "Slope < 10 deg  x  Illumination > 20%  x  Ice proximity  x  Terrain roughness",
            color="white", fontsize=13, y=0.97
        )

        plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="#080808")
        plt.show()
        print(f"    Saved: {out_png}")

    # ── STEP 8: Final summary ────────────────────────────────────
    print("\n[8/8] Final summary")
    print("=" * 64)
    print("LANDING SITE SELECTION COMPLETE — SUMMARY")
    print("=" * 64)
    print(f"  Total landable area          : {landable.sum() * PIXEL_KM2:.2f} km^2")
    print(f"  Candidate site clusters      : {len(sites)}")

    if best:
        print()
        print("  RECOMMENDED PRIMARY LANDING SITE")
        print(f"  Rank          : #{best['rank']}")
        print(f"  Coordinates   : {best['centroid_lat']:.4f} deg N  {best['centroid_lon']:.4f} deg E")

    return {
        "landing_score": landing_score,
        "site_pixels": site_pixels,
        "landable": landable,
        "sites": sites,
        "best_site": best,
    }