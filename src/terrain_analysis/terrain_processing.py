"""
Standalone terrain-derivative rasters, tabular data export, and DEM
reference-info reporting.

Extracted from PSR_Mapping.ipynb cells 44 (slope map), 61 (hillshade),
52 (combined CSV export), and 58 (CRS/bounds reference info).

Note on slope: this module's ``compute_slope_gradient`` uses
``np.gradient``, matching cells 44/52/61 exactly. This is a DIFFERENT
computation from ``ice_trap_detection.compute_slope_sobel`` /
``landing_site_scoring``'s inline Sobel-based slope (cells 30/34) — the
two produce slightly different numeric results and are kept separate
rather than merged into one "canonical" slope function, since neither
notebook cell treats them as interchangeable.

No algorithm or parameter has been changed from the original notebook
cells. Surrounding I/O has been wrapped into callable functions with
explicit parameters in place of notebook globals and hardcoded Drive
paths.
"""

from typing import Optional

import numpy as np
import pandas as pd
import rasterio

from .utils import load_dem, save_raster


def compute_slope_gradient(dem: np.ndarray, cell_size: float = 100.0) -> np.ndarray:
    """
    Compute terrain slope (degrees) from a DEM using ``np.gradient``.

    Reproduced verbatim from PSR_Mapping.ipynb cells 44 and 52 (identical
    computation in both). See the module docstring for why this is kept
    separate from the Sobel-based slope used in ice-trap/landing-site
    scoring.

    Parameters
    ----------
    dem : np.ndarray
        DEM elevation array.
    cell_size : float, default 100.0
        DEM pixel size in metres (passed as the spacing argument to
        ``np.gradient``).

    Returns
    -------
    np.ndarray
        Slope in degrees, same shape as ``dem``.
    """
    dy, dx = np.gradient(dem, cell_size)
    slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))
    return slope


def save_slope_map(dem_path: str, output_path: str) -> np.ndarray:
    """
    Compute the gradient-based slope map for a DEM and save it as a
    GeoTIFF.

    Reproduces PSR_Mapping.ipynb cell 44 exactly (loads the DEM directly
    via rasterio rather than ``utils.load_dem``, matching the original
    cell's behavior of not NaN-masking nodata before computing slope).

    Parameters
    ----------
    dem_path : str
        Path to the DEM GeoTIFF.
    output_path : str
        Path to write ``slope_map.tif`` to.

    Returns
    -------
    np.ndarray
        The computed slope array (degrees).
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile

    dy, dx = np.gradient(dem, 100)
    slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

    profile.update(dtype="float32")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(slope.astype(np.float32), 1)

    print("Saved:", output_path)
    print("Shape:", slope.shape)
    print("Slope range:", slope.min(), "to", slope.max())

    return slope


def compute_hillshade(
    dem: np.ndarray,
    azimuth: float = 315,
    altitude: float = 45,
    cell_size: float = 100.0,
) -> np.ndarray:
    """
    Compute an analytical hillshade from a DEM.

    Reproduced verbatim from PSR_Mapping.ipynb cell 61: standard
    hillshade formula using sun azimuth/altitude, gradient-derived
    slope/aspect, output rescaled to the 0-255 range.

    Parameters
    ----------
    dem : np.ndarray
        DEM elevation array.
    azimuth : float, default 315
        Sun azimuth in degrees (compass direction the light comes from).
    altitude : float, default 45
        Sun altitude in degrees above the horizon.
    cell_size : float, default 100.0
        DEM pixel size in metres.

    Returns
    -------
    np.ndarray
        Hillshade values rescaled to [0, 255].
    """
    x, y = np.gradient(dem, cell_size)

    slope = np.pi / 2 - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)

    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)

    hillshade = (
        np.sin(alt_rad) * np.sin(slope)
        + np.cos(alt_rad) * np.cos(slope)
        * np.cos(az_rad - aspect)
    )

    hillshade = 255 * (hillshade + 1) / 2
    return hillshade


def save_hillshade(dem_path: str, output_path: str,
                   azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    """
    Compute the hillshade for a DEM and save it as a GeoTIFF.

    Reproduces PSR_Mapping.ipynb cell 61's I/O exactly (loads the DEM
    directly via rasterio, matching the original cell's behavior).

    Parameters
    ----------
    dem_path : str
        Path to the DEM GeoTIFF.
    output_path : str
        Path to write ``hillshade.tif`` to.
    azimuth : float, default 315
        Sun azimuth in degrees.
    altitude : float, default 45
        Sun altitude in degrees.

    Returns
    -------
    np.ndarray
        The computed hillshade array, rescaled to [0, 255].
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        profile = src.profile

    hillshade = compute_hillshade(dem, azimuth=azimuth, altitude=altitude, cell_size=100)

    profile.update(dtype="float32")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(hillshade.astype(np.float32), 1)

    print("Saved hillshade.tif")
    return hillshade


def export_combined_dataset(
    dem_path: str,
    illumination_path: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    Flatten the DEM, gradient-based slope, and illumination fraction into
    a single row-per-pixel table and save it as CSV.

    Reproduces PSR_Mapping.ipynb cell 52 exactly, including its own
    independent (``np.gradient``-based) slope computation — this is
    intentionally the same formula as ``compute_slope_gradient`` above,
    since cell 52 computed slope inline rather than calling cell
    44/61's version, but produces identical numeric results given the
    same DEM and cell size.

    Parameters
    ----------
    dem_path : str
        Path to the DEM GeoTIFF.
    illumination_path : str
        Path to ``illumination_fraction.npy``.
    output_csv : str
        Path to write the combined CSV to.

    Returns
    -------
    pd.DataFrame
        The combined dataframe (columns: X_m, Y_m, Elevation_m,
        Slope_deg, Illumination_fraction), one row per pixel.
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform

    illum = np.load(illumination_path)

    dy, dx = np.gradient(dem, 100)
    slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

    rows, cols = dem.shape
    r, c = np.indices((rows, cols))
    x, y = rasterio.transform.xy(transform, r, c)

    combined_df = pd.DataFrame({
        "X_m": np.array(x).flatten(),
        "Y_m": np.array(y).flatten(),
        "Elevation_m": dem.flatten(),
        "Slope_deg": slope.flatten(),
        "Illumination_fraction": illum.flatten(),
    })

    print(combined_df.head())

    combined_df.to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print("Rows:", len(combined_df))

    return combined_df


def print_dem_reference_info(dem_path: str) -> dict:
    """
    Print and return the DEM's coordinate reference system and spatial
    bounds.

    Reproduced verbatim from PSR_Mapping.ipynb cell 58 — useful for
    documentation and cross-referencing this DEM against other datasets
    (e.g. in QGIS).

    Parameters
    ----------
    dem_path : str
        Path to the DEM GeoTIFF.

    Returns
    -------
    dict
        ``{"crs": rasterio.crs.CRS, "bounds": rasterio.coords.BoundingBox}``.
    """
    with rasterio.open(dem_path) as src:
        print("CRS:")
        print(src.crs)
        print("\nBounds:")
        print(src.bounds)
        return {"crs": src.crs, "bounds": src.bounds}