"""Quick test of histogram visual functionality."""

import numpy as np

from napari._vispy.visuals.histogram import (
    HistogramVisual,
    _hist_counts_to_mesh,
)


def test_hist_counts_to_mesh():
    """Test mesh generation from histogram counts."""
    counts = np.array([10, 20, 15, 5])
    bin_edges = np.array([0, 1, 2, 3, 4])

    vertices, faces = _hist_counts_to_mesh(
        counts, bin_edges, orientation='vertical'
    )

    # Should have 4 vertices per bin (4 bins = 16 vertices)
    assert vertices.shape == (16, 3)
    # Should have 2 triangles per bin (4 bins = 8 faces)
    assert faces.shape == (8, 3)
    # All Z coordinates should be 0
    assert np.all(vertices[:, 2] == 0)


def test_histogram_visual_basic():
    """Test basic histogram visual creation and data setting."""
    hist = HistogramVisual()

    # Test with simple data
    data = np.random.randn(1000)
    hist.set_data(data, bins=50)

    # Check that histogram was computed
    assert hist._counts is not None
    assert hist._bin_edges is not None
    assert len(hist._counts) == 50
    assert len(hist._bin_edges) == 51  # n_bins + 1

    # Test with None data (should handle gracefully)
    hist.set_data(None)
    assert hist.visible is False


def test_histogram_clims():
    """Test contrast limit indicators."""
    hist = HistogramVisual()
    data = np.random.randn(1000)
    hist.set_data(data, bins=50)

    # Set contrast limits
    hist.set_clims((-1.0, 1.0))

    # Lines should be visible
    assert hist.clim_lines.visible is True


if __name__ == '__main__':
    test_hist_counts_to_mesh()
    test_histogram_visual_basic()
    test_histogram_clims()
