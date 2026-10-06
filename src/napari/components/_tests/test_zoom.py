"""Zoom overlay."""

import pytest
from pydantic import ValidationError

from napari.components.overlays.rectangle import SelectionRectOverlay


def test_zoom():
    """Test creating zoom object"""
    zoom = SelectionRectOverlay()
    assert zoom is not None


def test_zoom_values():
    """Test creating zoom object"""
    zoom = SelectionRectOverlay()
    # validate position
    zoom.corners_canvas = ((0, 0), (300, 200))

    # data_positions must be 2-D
    with pytest.raises(ValidationError):
        zoom.corners_canvas = ((0, 0, 0), (300, 200, 100))

    assert zoom.corners_canvas == ((0, 0), (300, 200))
