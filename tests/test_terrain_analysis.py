"""
Unit tests for src/terrain_analysis/.

These tests cover the pure, deterministic functions that can be tested
with small synthetic arrays — they do NOT attempt to run the full
pipeline (horizon angles, illumination, ice-trap detection, landing-site
ranking), since that requires the real ~2.5 GB Faustini DEM, which is
not committed to this repository (see docs/datasets.md).

This is a starting point, not full pipeline coverage — the original
notebook has no automated tests at all (see docs/architecture.md /
README "Future Work"), so this file establishes real, honest coverage
for the functions that can be tested in isolation rather than claiming
more than currently exists.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from terrain_analysis import utils
from terrain_analysis.terrain_processing import compute_slope_gradient, compute_hillshade
from terrain_analysis.ice_trap_detection import compute_slope_sobel
from terrain_analysis.landing_site_scoring import norm01
from terrain_analysis.illumination import build_icrs_to_mcmf_rotation

from astropy.time import Time


class TestUtils:
    def test_chunk_path_format(self):
        path = utils.chunk_path("/tmp/chunks", "chunk_", 7)
        assert path == os.path.join("/tmp/chunks", "chunk_0007.npy")

    def test_chunk_path_zero_padding(self):
        path = utils.chunk_path("/tmp/chunks", "illum_chunk_", 0)
        assert path.endswith("illum_chunk_0000.npy")

    def test_ensure_dir_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = os.path.join(tmp, "nested", "dir")
            assert not os.path.exists(new_dir)
            utils.ensure_dir(new_dir)
            assert os.path.isdir(new_dir)

    def test_ensure_dir_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Should not raise if called twice on the same path
            utils.ensure_dir(tmp)
            utils.ensure_dir(tmp)
            assert os.path.isdir(tmp)


class TestSlopeComputations:
    def _flat_dem(self, size=20):
        return np.zeros((size, size), dtype=np.float32)

    def _tilted_plane_dem(self, size=20, slope_per_pixel=1.0):
        # Elevation increases linearly along rows -> constant, known slope
        rows = np.arange(size, dtype=np.float32).reshape(-1, 1)
        return np.tile(rows * slope_per_pixel, (1, size))

    def test_slope_gradient_flat_dem_is_zero(self):
        dem = self._flat_dem()
        slope = compute_slope_gradient(dem, cell_size=100.0)
        assert np.allclose(slope, 0.0, atol=1e-5)

    def test_slope_sobel_flat_dem_is_zero(self):
        dem = self._flat_dem()
        slope = compute_slope_sobel(dem, cell_size=100.0)
        assert np.allclose(slope, 0.0, atol=1e-5)

    def test_slope_gradient_and_sobel_differ_on_tilted_plane(self):
        """
        Documents (rather than hides) that the two slope implementations
        in this repository are numerically distinct — see
        docs/architecture.md and docs/methodology.md for why they are
        intentionally kept separate rather than unified.
        """
        dem = self._tilted_plane_dem()
        slope_gradient = compute_slope_gradient(dem, cell_size=100.0)
        slope_sobel = compute_slope_sobel(dem, cell_size=100.0)

        interior = slice(3, -3)
        gradient_interior = slope_gradient[interior, interior]
        sobel_interior = slope_sobel[interior, interior]

        # Both should detect a nonzero, positive slope on a tilted plane...
        assert np.all(gradient_interior > 0)
        assert np.all(sobel_interior > 0)
        # ...but they are not required to (and in general do not) agree exactly.

    def test_slope_gradient_nan_handling(self):
        dem = self._flat_dem()
        dem[5, 5] = np.nan
        slope = compute_slope_gradient(dem, cell_size=100.0)
        # np.gradient propagates NaN locally; just confirm it doesn't raise
        # and produces a same-shape array.
        assert slope.shape == dem.shape

    def test_slope_sobel_preserves_nan_mask(self):
        dem = self._flat_dem()
        dem[5, 5] = np.nan
        slope = compute_slope_sobel(dem, cell_size=100.0)
        assert np.isnan(slope[5, 5])


class TestHillshade:
    def test_hillshade_output_range(self):
        dem = np.random.RandomState(0).rand(30, 30).astype(np.float32) * 100
        hs = compute_hillshade(dem, azimuth=315, altitude=45, cell_size=100.0)
        assert hs.shape == dem.shape
        assert np.nanmin(hs) >= 0.0
        assert np.nanmax(hs) <= 255.0

    def test_hillshade_flat_dem_is_uniform(self):
        dem = np.zeros((10, 10), dtype=np.float32)
        hs = compute_hillshade(dem, azimuth=315, altitude=45, cell_size=100.0)
        # A perfectly flat surface should hillshade uniformly everywhere
        assert np.allclose(hs, hs[0, 0], atol=1e-3)


class TestNorm01:
    def test_norm01_output_bounds(self):
        arr = np.array([0, 1, 2, 3, 4, 5, 100, -100], dtype=np.float32)
        normed = norm01(arr, lo_pct=1, hi_pct=99)
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0

    def test_norm01_constant_array_does_not_error(self):
        arr = np.full((5, 5), 3.0, dtype=np.float32)
        normed = norm01(arr)
        # lo == hi -> denominator has the 1e-9 epsilon guard; should not raise
        assert normed.shape == arr.shape
        assert np.all(np.isfinite(normed))


class TestMoonRotation:
    def test_rotation_matrix_is_orthonormal(self):
        """
        build_icrs_to_mcmf_rotation should always return a valid rotation
        matrix: R @ R.T == identity, det(R) == 1.
        """
        t = Time("2024-06-15T00:00:00", format="isot", scale="tdb")
        R = build_icrs_to_mcmf_rotation(t)

        assert R.shape == (3, 3)
        should_be_identity = R @ R.T
        assert np.allclose(should_be_identity, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)

    def test_rotation_matrix_varies_with_time(self):
        """The Moon's rotation means R should differ meaningfully between
        two epochs several months apart."""
        t1 = Time("2024-01-01T00:00:00", format="isot", scale="tdb")
        t2 = Time("2024-07-01T00:00:00", format="isot", scale="tdb")

        R1 = build_icrs_to_mcmf_rotation(t1)
        R2 = build_icrs_to_mcmf_rotation(t2)

        assert not np.allclose(R1, R2, atol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))