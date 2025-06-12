"""Zoom overlay."""

import pytest

from napari._pydantic_compat import ValidationError
from napari.components.overlays.zoom import ZoomOverlay


def test_zoom():
    """Test creating zoom object"""
    zoom = ZoomOverlay()
    assert zoom is not None


def test_zoom_values():
    """Test creating zoom object"""
    zoom = ZoomOverlay()
    # validate canvas_positions
    zoom.canvas_positions = ((0, 0), (300, 200))
    with pytest.raises(ValidationError):
        # data_positions must be 3-D
        zoom.canvas_positions = ((0, 0, 0), (300, 200, 100))

    # validate data_positions
    zoom.data_positions = ((0, 0), (300, 200))
    mins, maxs = zoom.data_extents((0, 1))
    assert mins.shape == maxs.shape == (2,)

    zoom.data_positions = ((0, 0, 0), (300, 200, 200))
    mins, maxs = zoom.data_extents((0, 1, 2))
    assert mins.shape == maxs.shape == (3,)

    zoom.data_positions = ((0, 0, 0, 0), (300, 200, 200, 400))
    mins, maxs = zoom.data_extents((0, 1, 2, 3))
    assert mins.shape == maxs.shape == (4,)
