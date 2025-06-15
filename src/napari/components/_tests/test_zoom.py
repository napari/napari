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
