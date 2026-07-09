# """
# Generic I/O and path-handling helpers shared across the terrain_analysis
# package.

# This module intentionally contains no scientific logic — only the
# repeated raster I/O and chunk-file path patterns that appear near-identically
# across multiple stages of the LIRAF terrain-analysis pipeline
# (PSR_Mapping.ipynb, cells 7, 10, 23, 44, 52, 58, 61).
# """

# import os
# from typing import Optional

# import numpy as np
# import rasterio


# def load_dem(path: str) -> dict:
#     """
#     Load a DEM GeoTIFF and return its array together with the metadata
#     needed by downstream terrain-analysis stages.

#     This consolidates a raster-loading pattern that was repeated nearly
#     verbatim across multiple notebook cells (e.g. PSR_Mapping.ipynb cells
#     7, 44, 52, 58, 61), each of which opened the DEM with `rasterio.open`,
#     read band 1 as float32, and masked nodata pixels to NaN. No part of
#     that logic has been changed here — only consolidated into one function.

#     Parameters
#     ----------
#     path : str
#         Path to the DEM GeoTIFF (e.g. a LOLA-derived DEM such as
#         ``faustini_test_dem_100m.tif``).

#     Returns
#     -------
#     dict
#         Dictionary with keys:
#         - ``dem`` : np.ndarray (float32), nodata pixels set to NaN
#         - ``profile`` : dict, the rasterio profile (for writing derived rasters)
#         - ``transform`` : affine.Affine, the raster's geotransform
#         - ``crs`` : rasterio.crs.CRS
#         - ``bounds`` : rasterio.coords.BoundingBox
#         - ``nodata`` : float or None, the original nodata value
#     """
#     with rasterio.open(path) as src:
#         dem = src.read(1).astype(np.float32)
#         profile = src.profile.copy()
#         transform = src.transform
#         crs = src.crs
#         bounds = src.bounds
#         nodata = src.nodata

#     if nodata is not None:
#         dem[dem == nodata] = np.nan

#     return {
#         "dem": dem,
#         "profile": profile,
#         "transform": transform,
#         "crs": crs,
#         "bounds": bounds,
#         "nodata": nodata,
#     }


# def save_raster(path: str, array: np.ndarray, profile: dict) -> None:
#     """
#     Write a single-band array to a GeoTIFF using a rasterio profile.

#     Consolidates the raster-writing pattern repeated in PSR_Mapping.ipynb
#     cells 44 (slope map) and 61 (hillshade): update the profile's dtype to
#     match the array being written, then write it as band 1. Behavior is
#     unchanged from the original cells.

#     Parameters
#     ----------
#     path : str
#         Output GeoTIFF path.
#     array : np.ndarray
#         Single-band raster to write (will be cast to float32, matching
#         the original cells).
#     profile : dict
#         A rasterio profile (typically obtained from ``load_dem``), whose
#         ``dtype`` will be updated to ``"float32"`` before writing.
#     """
#     out_profile = profile.copy()
#     out_profile.update(dtype="float32")

#     with rasterio.open(path, "w", **out_profile) as dst:
#         dst.write(array.astype(np.float32), 1)


# def chunk_path(chunk_dir: str, prefix: str, index: int) -> str:
#     """
#     Build the file path for a numbered chunk file.

#     This generalizes two near-identical local closures found in the
#     original notebook:
#       - PSR_Mapping.ipynb cell 10 (horizon-angle chunks):
#             os.path.join(CHUNK_DIR, f"chunk_{ch_idx:04d}.npy")
#       - PSR_Mapping.ipynb cell 23 (illumination chunks):
#             os.path.join(CHUNK_DIR, f"illum_chunk_{ch:04d}.npy")

#     Both had identical structure and differed only in the filename
#     prefix ("chunk_" vs "illum_chunk_"). This function reproduces that
#     exact naming pattern with the prefix exposed as a parameter, so no
#     file-naming behavior changes for either call site as long as the
#     original prefixes are passed in (see horizon_angles.py and
#     illumination.py for the exact prefixes used).

#     Parameters
#     ----------
#     chunk_dir : str
#         Directory containing the chunk files.
#     prefix : str
#         Filename prefix (e.g. ``"chunk_"`` or ``"illum_chunk_"``).
#     index : int
#         Chunk index, zero-padded to 4 digits in the filename.

#     Returns
#     -------
#     str
#         Full path to the chunk file, e.g. ``"{chunk_dir}/{prefix}0007.npy"``.
#     """
#     return os.path.join(chunk_dir, f"{prefix}{index:04d}.npy")


# def ensure_dir(path: str) -> None:
#     """
#     Create a directory if it does not already exist.

#     Thin wrapper around ``os.makedirs(path, exist_ok=True)``, used
#     throughout the original notebook (e.g. cells 5, 10, 17, 23) to
#     prepare output directories before writing.

#     Parameters
#     ----------
#     path : str
#         Directory path to create.
#     """
#     os.makedirs(path, exist_ok=True)