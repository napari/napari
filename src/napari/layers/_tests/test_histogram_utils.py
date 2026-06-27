"""Tests for pure histogram utility functions.

These tests do not require Qt and can run in headless mode.
"""

from __future__ import annotations

import numpy as np

from napari.layers._histogram_utils import (
    auto_bins,
    compute_histogram,
    crop_to_range,
    downsample_histogram,
    log_transform,
)


class TestComputeHistogram:
    """Tests for compute_histogram()."""

    def test_float_data(self):
        data = np.random.rand(1000).astype(np.float32)
        edges, counts = compute_histogram(data, n_bins=10)
        assert len(edges) == 11  # 10 bins = 11 edges
        assert len(counts) == 10
        assert counts.sum() > 0

    def test_uint8_data(self):
        data = np.arange(256, dtype=np.uint8)
        _, counts = compute_histogram(data, n_bins=256)
        assert len(counts) == 256
        assert counts.sum() > 0

    def test_uint16_data(self):
        data = np.arange(1000, dtype=np.uint16)
        _, counts = compute_histogram(data, n_bins=50)
        assert len(counts) == 50
        assert counts.sum() > 0

    def test_all_ones(self):
        data = np.ones(100, dtype=np.float32)
        edges, counts = compute_histogram(data, n_bins=10)
        assert len(edges) == 11
        assert len(counts) == 10

    def test_empty_data(self):
        data = np.array([], dtype=np.float32)
        edges, counts = compute_histogram(data)
        assert len(edges) == 2
        assert len(counts) == 1

    def test_all_nan(self):
        data = np.full(100, np.nan)
        edges, counts = compute_histogram(data)
        assert len(edges) == 2
        assert len(counts) == 1

    def test_with_inf(self):
        data = np.array([1.0, 2.0, np.inf, -np.inf, 3.0])
        _, counts = compute_histogram(data, n_bins=5)
        assert len(counts) == 5
        assert counts.sum() > 0

    def test_custom_range(self):
        data = np.random.rand(1000) * 100
        edges, counts = compute_histogram(data, n_bins=10, range_=(0, 50))
        assert len(edges) == 11
        assert len(counts) == 10

    def test_negative_values(self):
        data = np.linspace(-10, 10, 1000)
        edges, counts = compute_histogram(data, n_bins=20)
        assert len(edges) == 21
        assert len(counts) == 20
        assert counts.sum() > 0


class TestLogTransform:
    """Tests for log_transform()."""

    def test_log10(self):
        counts = np.array([0, 1, 10, 100, 1000], dtype=np.float32)
        result = log_transform(counts, base=10.0)
        expected = np.log10(counts + 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_log_zero_counts(self):
        counts = np.array([0, 0, 0], dtype=np.float32)
        result = log_transform(counts)
        # log10(0 + 1) = 0
        assert np.all(result == 0.0)

    def test_log_natural(self):
        counts = np.array([0, 1, 2], dtype=np.float32)
        result = log_transform(counts, base=np.e)
        expected = np.log(counts + 1)
        np.testing.assert_array_almost_equal(result, expected)


class TestAutoBins:
    """Tests for auto_bins()."""

    def test_uint8(self):
        data = np.zeros(10, dtype=np.uint8)
        assert auto_bins(data) == 256

    def test_uint16(self):
        data = np.zeros(10, dtype=np.uint16)
        assert auto_bins(data) == 256  # capped at max_bins=256

    def test_float32(self):
        data = np.zeros(10, dtype=np.float32)
        assert auto_bins(data) == 256

    def test_float64(self):
        data = np.zeros(10, dtype=np.float64)
        assert auto_bins(data) == 256

    def test_int16(self):
        data = np.zeros(10, dtype=np.int16)
        assert auto_bins(data) == 256

    def test_from_dtype(self):
        assert auto_bins(np.dtype(np.uint8)) == 256
        assert auto_bins(np.dtype(np.float32)) == 256


class TestDownsampleHistogram:
    """Tests for downsample_histogram()."""

    def test_no_downsample_needed(self):
        counts = np.array([1, 2, 3, 4], dtype=np.float32)
        edges = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        d_counts, d_edges = downsample_histogram(counts, edges, max_bins=10)
        np.testing.assert_array_equal(d_counts, counts)
        np.testing.assert_array_equal(d_edges, edges)

    def test_downsample_10_to_5(self):
        counts = np.arange(1, 11, dtype=np.float32)
        edges = np.arange(0, 11, dtype=np.float32)
        d_counts, d_edges = downsample_histogram(counts, edges, max_bins=5)
        assert len(d_counts) <= 5
        assert len(d_edges) == len(d_counts) + 1

    def test_downsample_no_edges(self):
        counts = np.arange(1, 101, dtype=np.float32)
        d_counts, _ = downsample_histogram(counts, max_bins=10)
        assert len(d_counts) <= 10


class TestCropToRange:
    """Tests for crop_to_range()."""

    def test_crop_center(self):
        counts = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        edges = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)
        c_counts, c_edges = crop_to_range(counts, edges, (1, 4))
        assert len(c_counts) <= len(counts)
        assert len(c_edges) == len(c_counts) + 1

    def test_crop_outside_bounds(self):
        counts = np.array([1, 2, 3], dtype=np.float32)
        edges = np.array([0, 1, 2, 3], dtype=np.float32)
        c_counts, c_edges = crop_to_range(counts, edges, (-10, 10))
        np.testing.assert_array_equal(c_counts, counts)
        np.testing.assert_array_equal(c_edges, edges)

    def test_crop_invalid_range(self):
        """Crop with inverted range returns original."""
        counts = np.array([1, 2, 3], dtype=np.float32)
        edges = np.array([0, 1, 2, 3], dtype=np.float32)
        c_counts, c_edges = crop_to_range(counts, edges, (5, 3))
        np.testing.assert_array_equal(c_counts, counts)
        np.testing.assert_array_equal(c_edges, edges)
