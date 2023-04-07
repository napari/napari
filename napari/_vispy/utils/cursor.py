from __future__ import annotations

from enum import Enum

from qtpy.QtCore import QPoint, QSize, Qt
from qtpy.QtGui import QCursor, QPainter, QPen, QPixmap


def circle_pixmap(size: int):
    """Create a white/black hollow circle pixmap. For use as labels cursor."""
    size = max(size, 1)
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setPen(Qt.GlobalColor.white)
    painter.drawEllipse(0, 0, size - 1, size - 1)
    painter.setPen(Qt.GlobalColor.black)
    painter.drawEllipse(1, 1, size - 3, size - 3)
    painter.end()
    return pixmap


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


def square_pixmap(size):
    """Create a white/black hollow square pixmap. For use as labels cursor."""
    size = max(int(size), 1)
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setPen(Qt.GlobalColor.white)
    painter.drawRect(0, 0, size - 1, size - 1)
    painter.setPen(Qt.GlobalColor.black)
    painter.drawRect(1, 1, size - 3, size - 3)
    painter.end()
    return pixmap


def create_square_cursor(size):
    return QCursor(square_pixmap(size))


def create_circle_cursor(size):
    return QCursor(circle_pixmap(size))


def create_crosshair_cursor():
    return QCursor(crosshair_pixmap())


class QtCursorVisual(Enum):
    square = staticmethod(create_square_cursor)
    circle = staticmethod(create_circle_cursor)
    cross = Qt.CursorShape.CrossCursor
    forbidden = Qt.CursorShape.ForbiddenCursor
    pointing = Qt.CursorShape.PointingHandCursor
    standard = Qt.CursorShape.ArrowCursor
    crosshair = staticmethod(create_crosshair_cursor)
