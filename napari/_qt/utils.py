from contextlib import contextmanager
from functools import lru_cache, partial
from typing import Sequence, Union

import numpy as np
import qtpy
from qtpy.QtCore import QByteArray, QPropertyAnimation, QSize, Qt
from qtpy.QtGui import QColor, QCursor, QDrag, QImage, QPainter, QPixmap
from qtpy.QtWidgets import (
    QGraphicsColorizeEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QListWidget,
    QVBoxLayout,
    QWidget,
)

from ..utils.colormaps.standardize_color import transform_color
from ..utils.events.custom_types import Array
from ..utils.misc import is_sequence
from ..utils.translations import trans

QBYTE_FLAG = "!QBYTE_"


def is_qbyte(string: str) -> bool:
    """Check if a string is a QByteArray string.

    Parameters
    ----------
    string : bool
        State string.
    """
    return isinstance(string, str) and string.startswith(QBYTE_FLAG)


def qbytearray_to_str(qbyte: QByteArray) -> str:
    """Convert a window state to a string.

    Used for restoring the state of the main window.

    Parameters
    ----------
    qbyte : QByteArray
        State array.
    """
    return QBYTE_FLAG + qbyte.toBase64().data().decode()


def str_to_qbytearray(string: str) -> QByteArray:
    """Convert a string to a QbyteArray.

    Used for restoring the state of the main window.

    Parameters
    ----------
    string : str
        State string.
    """
    if len(string) < len(QBYTE_FLAG) or not is_qbyte(string):
        raise ValueError(
            trans._(
                "Invalid QByte string. QByte strings start with '{QBYTE_FLAG}'",
                QBYTE_FLAG=QBYTE_FLAG,
            )
        )

    return QByteArray.fromBase64(string[len(QBYTE_FLAG) :].encode())


def QImg2array(img):
    """Convert QImage to an array.

    Parameters
    ----------
    img : qtpy.QtGui.QImage
        QImage to be converted.

    Returns
    -------
    arr : array
        Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
        upper-left corner of the rendered region.
    """
    # Fix when  image is provided in wrong format (ex. test on Azure pipelines)
    if img.format() != QImage.Format_ARGB32:
        img = img.convertToFormat(QImage.Format_ARGB32)
    b = img.constBits()
    h, w, c = img.height(), img.width(), 4

    # As vispy doesn't use qtpy we need to reconcile the differences
    # between the `QImage` API for `PySide2` and `PyQt5` on how to convert
    # a QImage to a numpy array.
    if qtpy.API_NAME == 'PySide2':
        arr = np.array(b).reshape(h, w, c)
    else:
        b.setsize(h * w * c)
        arr = np.frombuffer(b, np.uint8).reshape(h, w, c)

    # Format of QImage is ARGB32_Premultiplied, but color channels are
    # reversed.
    arr = arr[:, :, [2, 1, 0, 3]]
    return arr


@contextmanager
def qt_signals_blocked(obj):
    """Context manager to temporarily block signals from `obj`"""
    obj.blockSignals(True)
    yield
    obj.blockSignals(False)


@contextmanager
def event_hook_removed():
    """Context manager to temporarily remove the PyQt5 input hook"""
    from qtpy import QtCore

    if hasattr(QtCore, 'pyqtRemoveInputHook'):
        QtCore.pyqtRemoveInputHook()
    try:
        yield
    finally:
        if hasattr(QtCore, 'pyqtRestoreInputHook'):
            QtCore.pyqtRestoreInputHook()


def disable_with_opacity(obj, widget_list, enabled):
    """Set enabled state on a list of widgets. If not enabled, decrease opacity."""
    for widget_name in widget_list:
        widget = getattr(obj, widget_name)
        widget.setEnabled(enabled)
        op = QGraphicsOpacityEffect(obj)
        op.setOpacity(1 if enabled else 0.5)
        widget.setGraphicsEffect(op)


@lru_cache(maxsize=64)
def square_pixmap(size):
    """Create a white/black hollow square pixmap. For use as labels cursor."""
    size = max(int(size), 1)
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setPen(Qt.white)
    painter.drawRect(0, 0, size - 1, size - 1)
    painter.setPen(Qt.black)
    painter.drawRect(1, 1, size - 3, size - 3)
    painter.end()
    return pixmap


@lru_cache(maxsize=64)
def circle_pixmap(size: int):
    """Create a white/black hollow circle pixmap. For use as labels cursor."""
    size = max(int(size), 1)
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setPen(Qt.white)
    painter.drawEllipse(0, 0, size - 1, size - 1)
    painter.setPen(Qt.black)
    painter.drawEllipse(1, 1, size - 3, size - 3)
    painter.end()
    return pixmap


def drag_with_pixmap(list_widget: QListWidget) -> QDrag:
    """Create a QDrag object with a pixmap of the currently select list item.

    This method is useful when you have a QListWidget that displays custom
    widgets for each QListWidgetItem instance in the list (usually by calling
    ``QListWidget.setItemWidget(item, widget)``).  When used in a
    ``QListWidget.startDrag`` method, this function creates a QDrag object that
    shows an image of the item being dragged (rather than an empty rectangle).

    Parameters
    ----------
    list_widget : QListWidget
        The QListWidget for which to create a QDrag object.

    Returns
    -------
    QDrag
        A QDrag instance with a pixmap of the currently selected item.

    Examples
    --------
    >>> class QListWidget:
    ...     def startDrag(self, supportedActions):
    ...         drag = drag_with_pixmap(self)
    ...         drag.exec_(supportedActions, Qt.MoveAction)

    """
    drag = QDrag(list_widget)
    drag.setMimeData(list_widget.mimeData(list_widget.selectedItems()))
    size = list_widget.viewport().visibleRegion().boundingRect().size()
    pixmap = QPixmap(size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    for index in list_widget.selectedIndexes():
        rect = list_widget.visualRect(index)
        painter.drawPixmap(rect, list_widget.viewport().grab(rect))
    painter.end()
    drag.setPixmap(pixmap)
    drag.setHotSpot(list_widget.viewport().mapFromGlobal(QCursor.pos()))
    return drag


def combine_widgets(
    widgets: Union[QWidget, Sequence[QWidget]], vertical: bool = False
) -> QWidget:
    """Combine a list of widgets into a single QWidget with Layout.

    Parameters
    ----------
    widgets : QWidget or sequence of QWidget
        A widget or a list of widgets to combine.
    vertical : bool, optional
        Whether the layout should be QVBoxLayout or not, by default
        QHBoxLayout is used

    Returns
    -------
    QWidget
        If ``widgets`` is a sequence, returns combined QWidget with `.layout`
        property, otherwise returns the original widget.

    Raises
    ------
    TypeError
        If ``widgets`` is neither a ``QWidget`` or a sequence of ``QWidgets``.
    """
    if isinstance(getattr(widgets, 'native', None), QWidget):
        # compatibility with magicgui v0.2.0 which no longer uses QWidgets
        # directly. Like vispy, the backend widget is at widget.native
        return widgets.native  # type: ignore
    elif isinstance(widgets, QWidget):
        return widgets
    elif is_sequence(widgets):
        # the same as above, compatibility with magicgui v0.2.0
        widgets = [
            i.native if isinstance(getattr(i, 'native', None), QWidget) else i
            for i in widgets
        ]
        if all(isinstance(i, QWidget) for i in widgets):
            container = QWidget()
            container.setLayout(QVBoxLayout() if vertical else QHBoxLayout())
            for widget in widgets:
                container.layout().addWidget(widget)
            return container
    raise TypeError(
        trans._('"widget" must be a QWidget or a sequence of QWidgets')
    )


def add_flash_animation(
    widget: QWidget, duration: int = 300, color: Array = (0.5, 0.5, 0.5, 0.5)
):
    """Add flash animation to widget to highlight certain action (e.g. taking a screenshot).

    Parameters
    ----------
    widget : QWidget
        Any Qt widget.
    duration : int
        Duration of the flash animation.
    color : Array
        Color of the flash animation. By default, we use light gray.
    """
    color = transform_color(color)[0]
    color = (255 * color).astype("int")

    effect = QGraphicsColorizeEffect(widget)
    widget.setGraphicsEffect(effect)

    widget._flash_animation = QPropertyAnimation(effect, b"color")
    widget._flash_animation.setStartValue(QColor(0, 0, 0, 0))
    widget._flash_animation.setEndValue(QColor(0, 0, 0, 0))
    widget._flash_animation.setLoopCount(1)

    # let's make sure to remove the animation from the widget because
    # if we don't, the widget will actually be black and white.
    widget._flash_animation.finished.connect(
        partial(remove_flash_animation, widget)
    )

    widget._flash_animation.start()

    # now  set an actual time for the flashing and an intermediate color
    widget._flash_animation.setDuration(duration)
    widget._flash_animation.setKeyValueAt(0.1, QColor(*color))


def remove_flash_animation(widget: QWidget):
    """Remove flash animation from widget.

    Parameters
    ----------
    widget : QWidget
        Any Qt widget.
    """
    widget.setGraphicsEffect(None)
    del widget._flash_animation
