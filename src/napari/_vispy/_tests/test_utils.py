import pytest
from pydantic import ValidationError
from qtpy.QtCore import Qt

from napari._vispy.utils.cursor import QtCursorVisual
from napari.components._viewer_constants import CursorStyle


def test_set_cursor(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.cursor.style = CursorStyle.SQUARE.value
    viewer.cursor.size = 10
    assert (
        viewer.window._qt_viewer.canvas.cursor.shape()
        == Qt.CursorShape.BitmapCursor
    )

    viewer.cursor.size = 5
    assert (
        viewer.window._qt_viewer.canvas.cursor.shape()
        == QtCursorVisual['cross'].value
    )

    viewer.cursor.style = CursorStyle.CIRCLE.value
    viewer.cursor.size = 100

    assert viewer.canvas.overlays._brush_circle.visible
    assert viewer.canvas.overlays._brush_circle.size == viewer.cursor.size

    with pytest.raises(ValidationError):
        viewer.cursor.style = 'invalid'
