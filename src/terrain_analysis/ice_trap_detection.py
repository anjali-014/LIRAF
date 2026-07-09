"""
Candidate ice-trap detection within the PSR mask, Faustini crater.

Extracted from PSR_Mapping.ipynb, "CELL 9 — Candidate Ice Trap Detection
Inside PSRs". Scientific basis: Mazarico et al. (2011) [PSR definition],
Paige et al. (2010) [thermal cold-trap criteria], Hayne et al. (2015)
[cold-trap modeling], Rubanenko et al. (2019) [micro cold-trap
proximity], Watson et al. (1961) [classical cold-trap theory].

No algorithm, weight, or threshold has been changed from the original
notebook cell. Surrounding I/O/plotting/save logic has been wrapped into
a single callable function with explicit parameters in place of notebook
globals and hardcoded Drive paths.
"""

import os
import json
from typing import List, Dict, Optional

import numpy as np
import rasterio
from rasterio.transform import xy as rio_xy
from scipy.ndimage import (
    uniform_filter, label, maximum_filter, sobel,
)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from .utils import ensure_dir

# Pixel geometry (100 m/pixel DEM, matching the rest of the pipeline)
PIXEL_SIZE_M = 100
PIXEL_AREA_KM2 = (PIXEL_SIZE_M / 1000) ** 2

# Scoring weights (must sum to 1.0) — reproduced from cell 30
W_ILLUM = 0.30
W_DEPTH = 0.25
W_SLOPE = 0.20
W_RIM_DEPTH = 0.25

# Minimum connected-component area to report as a candidate region (km^2)
MIN_CLUSTER_AREA_KM2 = 0.5


def compute_slope_sobel(dem_arr: np.ndarray, cell_size: float = 100.0) -> np.ndarray:
    """
    Compute terrain slope (degrees) from a DEM using Sobel gradient
    operators.

    Reproduced verbatim from PSR_Mapping.ipynb cell 30 (there named
    ``compute_slope_deg``). Renamed here to ``compute_slope_sobel`` to
    disambiguate it from ``terrain_processing.compute_slope_gradient``,
    which computes slope from the same DEM using ``np.gradient`` instead
    of Sobel operators (cells 44/47/49/52). The two produce slightly
    different numeric results and are used for different purposes in
    this pipeline (this Sobel version feeds ice-trap and landing-site
    scoring; the gradient version produces the standalone
    ``slope_map.tif`` terrain output) — they are intentionally NOT
    merged into one "canonical" slope function.

    Parameters
    ----------
    dem_arr : np.ndarray
        DEM elevation array (NaN for nodata).
    cell_size : float, default 100.0
        DEM pixel size in metres.

    Returns
    -------
    np.ndarray
        Slope in degrees, same shape as ``dem_arr``, NaN where the input
        DEM was NaN.
    """
    dem_f = np.where(np.isnan(dem_arr), 0.0, dem_arr)
    dz_dx = sobel(dem_f, axis=1) / (8.0 * cell_size)
    dz_dy = sobel(dem_f, axis=0) / (8.0 * cell_size)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
    slope[np.isnan(dem_arr)] = np.nan
    return slope.astype(np.float32)


def label_ax(ax, title: str, subtitle: str = "") -> None:
    """
    Apply consistent dark-theme title/subtitle styling to a Matplotlib
    axis.

    Reproduced verbatim from PSR_Mapping.ipynb cell 30. Note: a
    differently-implemented axis-styling helper of the same conceptual
    purpose also exists in ``landing_site_scoring.style_ax`` (from cell
    34) — the two are kept separate rather than merged, since they were
    defined independently in the original notebook with slightly
    different styling details.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style.
    title : str
        Main title text.
    subtitle : str, default ""
        Optional subtitle text, drawn above the axis in a smaller font.
    """
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", color="#aaaaaa", fontsize=8)


def detect_candidate_ice_traps(
    dem_path: str,
    psr_mask_path: str,
    illumination_path: str,
    output_dir: str,
    make_diagnostic_plot: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Detect and rank candidate subsurface ice-trap regions within the PSR
    mask, using a weighted composite of illumination, local topographic
    depression, slope, and depth below the enclosing crater rim.

    Reproduces the full orchestration of PSR_Mapping.ipynb cell 30:

    1. Load DEM, PSR mask, illumination fraction.
    2. Compute four 0-1 normalized scoring layers:
       - illumination score (``1 - illumination_fraction``)
       - local depression score (pixel vs. 500 m neighborhood mean)
       - slope score (flat=1, penalty starts above 10 deg, zero above 30 deg)
       - rim-depth score (depth below the max elevation within 20 km,
         a regional cold-trap proxy)
    3. Composite score = weighted sum (weights: illum 0.30, depth 0.25,
       slope 0.20, rim depth 0.25), hard-gated to zero outside the PSR.
    4. Threshold at the 80th/90th percentile of in-PSR scores to get
       candidate / high-confidence masks.
    5. Connected-component label the candidate mask (8-connectivity),
       drop regions smaller than 0.5 km^2, compute per-region
       lat/lon centroid and statistics, rank by mean score.
    6. Save the 3-layer array, the JSON region catalogue, a resume
       checkpoint, and (optionally) a 6-panel diagnostic figure.

    No weight, threshold, or formula differs from cell 30.

    Parameters
    ----------
    dem_path : str
        Path to the DEM GeoTIFF.
    psr_mask_path : str
        Path to ``psr_mask.npy`` (from ``illumination.compute_illumination_and_psr``).
    illumination_path : str
        Path to ``illumination_fraction.npy``.
    output_dir : str
        Directory to save ``candidate_ice_traps.npy``,
        ``candidate_ice_trap_catalogue.json``, ``cell9_checkpoint.npz``,
        and the diagnostic figure into.
    make_diagnostic_plot : bool, default True
        Whether to generate and save the 6-panel diagnostic figure.

    Returns
    -------
    dict
        ``{"composite": np.ndarray, "candidate_mask": np.ndarray (bool),
        "high_conf_mask": np.ndarray (bool), "regions": list[dict]}``.
    """
    ensure_dir(output_dir)
    checkpoint_path = os.path.join(output_dir, "cell9_checkpoint.npz")
    out_npy = os.path.join(output_dir, "candidate_ice_traps.npy")
    out_png = os.path.join(output_dir, "09_candidate_ice_traps.png")
    catalogue_path = os.path.join(output_dir, "candidate_ice_trap_catalogue.json")

    print("=" * 62)
    print("Candidate Ice Trap Detection")
    print("=" * 62)

    # ── STEP 1: Load data ────────────────────────────────────────
    print("\n[1/7] Loading inputs...")

    psr_mask = np.load(psr_mask_path).astype(bool)
    illum = np.load(illumination_path).astype(np.float32)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if nodata is not None:
        dem[dem == nodata] = np.nan

    rows, cols = dem.shape
    print(f"    DEM shape      : {rows} x {cols}")
    print(f"    PSR pixels     : {psr_mask.sum():,}  ({psr_mask.mean() * 100:.1f}% of DEM)")
    print(f"    PSR area       : {psr_mask.sum() * PIXEL_AREA_KM2:.1f} km^2")
    print(f"    Illum range    : {np.nanmin(illum):.4f} - {np.nanmax(illum):.4f}")

    # ── STEP 2: Compute scoring layers ───────────────────────────
    print("\n[2/7] Computing scoring layers...")

    illum_score = 1.0 - illum

    slope_deg = compute_slope_sobel(dem, PIXEL_SIZE_M)
    slope_score = np.clip(1.0 - (slope_deg / 30.0), 0.0, 1.0).astype(np.float32)

    radius_px = 5
    neighbourhood_mean = uniform_filter(
        np.where(np.isnan(dem), np.nanmean(dem), dem),
        size=2 * radius_px + 1
    ).astype(np.float32)

    local_relief = neighbourhood_mean - dem
    local_relief[np.isnan(dem)] = np.nan

    lr_min = np.nanpercentile(local_relief, 1)
    lr_max = np.nanpercentile(local_relief, 99)
    depression_score = np.clip(
        (local_relief - lr_min) / (lr_max - lr_min + 1e-9), 0.0, 1.0
    ).astype(np.float32)

    rim_radius_px = 200
    rim_elevation = maximum_filter(
        np.where(np.isnan(dem), np.nanmin(dem), dem),
        size=2 * rim_radius_px + 1
    ).astype(np.float32)

    depth_below_rim = rim_elevation - dem
    depth_below_rim[np.isnan(dem)] = np.nan

    dr_min = np.nanpercentile(depth_below_rim, 1)
    dr_max = np.nanpercentile(depth_below_rim, 99)
    rim_depth_score = np.clip(
        (depth_below_rim - dr_min) / (dr_max - dr_min + 1e-9), 0.0, 1.0
    ).astype(np.float32)

    print(f"    Slope range        : {np.nanmin(slope_deg):.1f} - {np.nanmax(slope_deg):.1f} deg")
    print(f"    Local relief range : {np.nanmin(local_relief):.1f} m - {np.nanmax(local_relief):.1f} m")
    print(f"    Rim depth range    : {np.nanmin(depth_below_rim):.1f} m - {np.nanmax(depth_below_rim):.1f} m")

    # ── STEP 3: Composite ice-trap score ─────────────────────────
    print("\n[3/7] Computing composite ice-trap score...")

    composite = (
        W_ILLUM * illum_score +
        W_DEPTH * depression_score +
        W_SLOPE * slope_score +
        W_RIM_DEPTH * rim_depth_score
    ).astype(np.float32)

    composite[~psr_mask] = 0.0
    composite[np.isnan(dem)] = 0.0

    print(f"    Composite range inside PSR: "
          f"{composite[psr_mask].min():.4f} - {composite[psr_mask].max():.4f}")
    print(f"    Mean score inside PSR     : {composite[psr_mask].mean():.4f}")

    # ── STEP 4: Threshold -> candidate mask ──────────────────────
    print("\n[4/7] Identifying candidate ice-trap pixels...")

    psr_scores = composite[psr_mask]
    threshold_80 = np.percentile(psr_scores, 80)
    threshold_90 = np.percentile(psr_scores, 90)

    candidate_mask = (composite >= threshold_80) & psr_mask
    high_conf_mask = (composite >= threshold_90) & psr_mask

    print(f"    Score threshold (top 20%) : {threshold_80:.4f}")
    print(f"    Score threshold (top 10%) : {threshold_90:.4f}")
    print(f"    Candidate pixels          : {candidate_mask.sum():,}")
    print(f"    Candidate area            : {candidate_mask.sum() * PIXEL_AREA_KM2:.2f} km^2")
    print(f"    High-confidence pixels    : {high_conf_mask.sum():,}")
    print(f"    High-confidence area      : {high_conf_mask.sum() * PIXEL_AREA_KM2:.2f} km^2")

    # ── STEP 5: Connected-component labelling + ranking ──────────
    print("\n[5/7] Labelling and ranking candidate regions...")

    structure = np.ones((3, 3), dtype=int)
    labeled_array, n_clusters = label(candidate_mask, structure=structure)

    print(f"    Total connected regions   : {n_clusters}")

    regions: List[Dict] = []
    for region_id in range(1, n_clusters + 1):
        region_pixels = labeled_array == region_id
        area_km2 = region_pixels.sum() * PIXEL_AREA_KM2

        if area_km2 < MIN_CLUSTER_AREA_KM2:
            continue

        region_scores = composite[region_pixels]
        region_illum = illum[region_pixels]
        region_elev = dem[region_pixels]
        region_slope = slope_deg[region_pixels]

        rr, cc = np.where(region_pixels)
        cx, cy = rio_xy(transform, rr.mean(), cc.mean())

        if crs and crs.is_geographic:
            lon, lat = cx, cy
        else:
            from pyproj import Transformer
            try:
                t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                lon, lat = t.transform(cx, cy)
            except Exception:
                lon, lat = cx, cy

        regions.append({
            "rank": 0,
            "region_id": int(region_id),
            "area_km2": round(float(area_km2), 3),
            "mean_score": round(float(region_scores.mean()), 4),
            "max_score": round(float(region_scores.max()), 4),
            "mean_illumination": round(float(region_illum.mean()), 6),
            "mean_elevation_m": round(float(np.nanmean(region_elev)), 1),
            "mean_slope_deg": round(float(np.nanmean(region_slope)), 2),
            "centroid_lat": round(float(lat), 4),
            "centroid_lon": round(float(lon), 4),
            "pixel_row": int(rr.mean()),
            "pixel_col": int(cc.mean()),
        })

    regions.sort(key=lambda x: (-x["mean_score"], -x["area_km2"]))
    for i, r in enumerate(regions):
        r["rank"] = i + 1

    print(f"    Regions above {MIN_CLUSTER_AREA_KM2} km^2  : {len(regions)}")

    # ── STEP 6: Save outputs ──────────────────────────────────────
    print("\n[6/7] Saving outputs...")

    np.save(out_npy, np.stack([
        composite,
        candidate_mask.astype(np.float32),
        high_conf_mask.astype(np.float32)
    ], axis=0))
    print(f"    Saved: {out_npy}")

    with open(catalogue_path, "w") as f:
        json.dump(regions, f, indent=2)
    print(f"    Saved: {catalogue_path}")

    np.savez(checkpoint_path,
             composite=composite,
             candidate_mask=candidate_mask,
             high_conf_mask=high_conf_mask)
    print(f"    Saved: {checkpoint_path}")

    # ── STEP 7: Diagnostic plots ─────────────────────────────────
    if make_diagnostic_plot:
        print("\n[7/7] Generating diagnostic figure...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="#0a0a0a")
        for ax in axes.flat:
            ax.set_facecolor("#0a0a0a")
            ax.tick_params(colors="#cccccc")
            for sp in ax.spines.values():
                sp.set_edgecolor("#444444")

        ax = axes[0, 0]
        dem_show = np.where(np.isnan(dem), np.nanmin(dem), dem)
        im = ax.imshow(dem_show, cmap="gist_earth", interpolation="nearest")
        psr_overlay = np.ma.masked_where(~psr_mask, np.ones_like(dem))
        ax.imshow(psr_overlay, cmap=mcolors.ListedColormap(["#3399ff"]),
                  alpha=0.45, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.03).set_label("Elevation (m)", color="#cccccc")
        label_ax(ax, "DEM + PSR mask", "blue = PSR")

        ax = axes[0, 1]
        illum_psr = np.ma.masked_where(~psr_mask, illum)
        im2 = ax.imshow(illum_psr, cmap="inferno", vmin=0, vmax=0.3,
                        interpolation="nearest")
        plt.colorbar(im2, ax=ax, fraction=0.03).set_label("Illumination fraction", color="#cccccc")
        label_ax(ax, "Illumination inside PSR", "lower = colder = better")

        ax = axes[0, 2]
        comp_psr = np.ma.masked_where(~psr_mask, composite)
        im3 = ax.imshow(comp_psr, cmap="plasma", vmin=0, vmax=1,
                        interpolation="nearest")
        plt.colorbar(im3, ax=ax, fraction=0.03).set_label("Ice-trap score", color="#cccccc")
        label_ax(ax, "Composite ice-trap score", "higher = more favourable")

        ax = axes[1, 0]
        ax.imshow(dem_show, cmap="gray", alpha=0.4, interpolation="nearest")
        tier1 = np.ma.masked_where(~candidate_mask | high_conf_mask, np.ones_like(dem))
        tier2 = np.ma.masked_where(~high_conf_mask, np.ones_like(dem))
        ax.imshow(tier1, cmap=mcolors.ListedColormap(["#ffaa00"]),
                  alpha=0.7, interpolation="nearest")
        ax.imshow(tier2, cmap=mcolors.ListedColormap(["#ff2255"]),
                  alpha=0.9, interpolation="nearest")

        for r in regions[:5]:
            ax.plot(r["pixel_col"], r["pixel_row"], "w*", markersize=9, zorder=5)
            ax.text(r["pixel_col"] + 8, r["pixel_row"],
                    f"#{r['rank']}", color="white", fontsize=7, zorder=6)

        legend_patches = [
            Patch(color="#ffaa00", label="Candidate (top 20%)"),
            Patch(color="#ff2255", label="High-confidence (top 10%)"),
        ]
        ax.legend(handles=legend_patches, loc="lower right",
                  facecolor="#1a1a1a", edgecolor="#444", labelcolor="white", fontsize=7)
        label_ax(ax, "Candidate ice-trap regions", "star = top-5 ranked sites")

        ax = axes[1, 1]
        psr_sc = composite[psr_mask]
        ax.hist(psr_sc, bins=80, color="#4488ff", edgecolor="none", alpha=0.85)
        ax.axvline(threshold_80, color="#ffaa00", lw=1.5, linestyle="--",
                   label=f"Top 20% threshold ({threshold_80:.3f})")
        ax.axvline(threshold_90, color="#ff2255", lw=1.5, linestyle="--",
                   label=f"Top 10% threshold ({threshold_90:.3f})")
        ax.set_xlabel("Composite score", color="#cccccc")
        ax.set_ylabel("Pixel count", color="#cccccc")
        ax.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="white", fontsize=8)
        label_ax(ax, "Score distribution within PSR")

        ax = axes[1, 2]
        ax.axis("off")
        top10 = regions[:10]
        if top10:
            col_labels = ["Rank", "Area km2", "Score", "Illum", "Slope deg", "Lat", "Lon"]
            table_data = [[
                str(r["rank"]),
                f"{r['area_km2']:.2f}",
                f"{r['mean_score']:.4f}",
                f"{r['mean_illumination']:.5f}",
                f"{r['mean_slope_deg']:.1f}",
                f"{r['centroid_lat']:.3f}",
                f"{r['centroid_lon']:.3f}",
            ] for r in top10]

            tbl = ax.table(cellText=table_data, colLabels=col_labels,
                          loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7.5)
            tbl.scale(1.0, 1.35)

            for (row_i, col_i), cell in tbl.get_celld().items():
                cell.set_facecolor("#1a1a1a" if row_i > 0 else "#2a2a2a")
                cell.set_edgecolor("#444444")
                cell.set_text_props(color="#eeeeee")

        label_ax(ax, "Top-10 candidate regions", "ranked by mean composite score")

        plt.suptitle(
            "Faustini Crater - Candidate Ice Trap Detection\n"
            "Mazarico 2011 PSR x Paige 2010 thermal criteria x Hayne 2015 cold-trap model",
            color="white", fontsize=12, y=1.01
        )
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
        plt.show()
        print(f"    Saved: {out_png}")

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("CANDIDATE ICE TRAP DETECTION COMPLETE — SUMMARY")
    print("=" * 62)
    print(f"  PSR area (Faustini)        : {psr_mask.sum() * PIXEL_AREA_KM2:.2f} km^2")
    print(f"  Candidate ice-trap area    : {candidate_mask.sum() * PIXEL_AREA_KM2:.2f} km^2")
    print(f"  High-confidence area       : {high_conf_mask.sum() * PIXEL_AREA_KM2:.2f} km^2")
    print(f"  Candidate / PSR ratio      : "
          f"{candidate_mask.sum() / max(psr_mask.sum(), 1) * 100:.1f}%")
    print(f"  Total candidate regions    : {len(regions)}")

    return {
        "composite": composite,
        "candidate_mask": candidate_mask,
        "high_conf_mask": high_conf_mask,
        "regions": regions,
    }