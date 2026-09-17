from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from qtpy.QtCore import QPoint, QSize, Qt
from qtpy.QtGui import QCursor, QPainter, QPen, QPixmap

from napari.layers.labels._labels_constants import Mode as LabelsMode
from napari.layers.labels.labels import Labels
from napari.layers.points._points_constants import Mode as PointsMode
from napari.layers.points.points import Points
from napari.layers.shapes._shapes_constants import Mode as ShapesMode
from napari.layers.shapes.shapes import Shapes

if TYPE_CHECKING:
    from napari.layers.base import Layer


def crosshair_pixmap():
    """Create a cross cursor with white/black hollow square pixmap in the middle.
    For use as points cursor."""

    size = 25

    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)

    # Base measures
    width = 1
    center = 3  # Must be odd!
    rect_size = center + 2 * width
    square = rect_size + width * 4

    pen = QPen(Qt.GlobalColor.white, 1)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.setPen(pen)

    # # Horizontal rectangle
    painter.drawRect(0, (size - rect_size) // 2, size - 1, rect_size - 1)

    # Vertical rectangle
    painter.drawRect((size - rect_size) // 2, 0, rect_size - 1, size - 1)

    # Square
    painter.drawRect(
        (size - square) // 2, (size - square) // 2, square - 1, square - 1
    )

    pen = QPen(Qt.GlobalColor.black, 2)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.setPen(pen)

    # # Square
    painter.drawRect(
        (size - square) // 2 + 2,
        (size - square) // 2 + 2,
        square - 4,
        square - 4,
    )

    pen = QPen(Qt.GlobalColor.black, 3)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.setPen(pen)

    # # # Horizontal lines
    mid_vpoint = QPoint(2, size // 2)
    painter.drawLine(
        mid_vpoint, QPoint(((size - center) // 2) - center + 1, size // 2)
    )
    mid_vpoint = QPoint(size - 3, size // 2)
    painter.drawLine(
        mid_vpoint, QPoint(((size - center) // 2) + center + 1, size // 2)
    )

    # # # Vertical lines
    mid_hpoint = QPoint(size // 2, 2)
    painter.drawLine(
        QPoint(size // 2, ((size - center) // 2) - center + 1), mid_hpoint
    )
    mid_hpoint = QPoint(size // 2, size - 3)
    painter.drawLine(
        QPoint(size // 2, ((size - center) // 2) + center + 1), mid_hpoint
    )

    painter.end()
    return pixmap


def create_crosshair_cursor():
    return QCursor(crosshair_pixmap())


def create_blank_cursor():
    return QCursor(Qt.CursorShape.BlankCursor)


class QtCursorVisual(Enum):
    blank = staticmethod(create_blank_cursor)
    pointing = Qt.CursorShape.PointingHandCursor
    standard = Qt.CursorShape.ArrowCursor
    crosshair = staticmethod(create_crosshair_cursor)


# cursor style is determined by the active layer and its mode.
# Omitted layers and modes fall back to 'standard'
_CURSOR_STYLES: dict[type, dict] = {
    Labels: {
        LabelsMode.PAINT: 'circle',
        LabelsMode.ERASE: 'circle',
        LabelsMode.FILL: 'crosshair',
        LabelsMode.POLYGON: 'crosshair',
        LabelsMode.PICK: 'pointing',
    },
    Points: {
        PointsMode.ADD: 'crosshair',
        PointsMode.SELECT: 'pointing',
    },
    Shapes: {
        ShapesMode.VERTEX_INSERT: 'crosshair',
        ShapesMode.VERTEX_REMOVE: 'crosshair',
        ShapesMode.ADD_RECTANGLE: 'crosshair',
        ShapesMode.ADD_ELLIPSE: 'crosshair',
        ShapesMode.ADD_LINE: 'crosshair',
        ShapesMode.ADD_POLYLINE: 'crosshair',
        ShapesMode.ADD_PATH: 'crosshair',
        ShapesMode.ADD_POLYGON: 'crosshair',
        ShapesMode.ADD_POLYGON_LASSO: 'crosshair',
        ShapesMode.SELECT: 'pointing',
        ShapesMode.DIRECT: 'pointing',
    },
}


def get_cursor_style(layer: Layer) -> str:
    """Return the cursor style string for the given layer and its mode.

    The cursor style is a viewer-side concept: it is derived from the active
    layer's mode and does not live on the layer itself.
    """
    for layer_cls, mode_cursors in _CURSOR_STYLES.items():
        if isinstance(layer, layer_cls):
            return mode_cursors.get(layer.mode, 'standard')
    return 'standard'
