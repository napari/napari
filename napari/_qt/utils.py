from __future__ import annotations

import re
import signal
import socket
import weakref
from contextlib import contextmanager
from functools import lru_cache, partial
from typing import Sequence, Union

import numpy as np
import qtpy
from qtpy.QtCore import (
    QByteArray,
    QObject,
    QPoint,
    QPropertyAnimation,
    QSize,
    QSocketNotifier,
    Qt,
    Signal,
)
from qtpy.QtGui import QColor, QCursor, QDrag, QImage, QPainter, QPen, QPixmap
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
RICH_TEXT_PATTERN = re.compile("<[^\n]+>")


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
    previous = obj.blockSignals(True)
    try:
        yield
    finally:
        obj.blockSignals(previous)


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
def crosshair_pixmap():
    """Create a cross cursor with white/black hollow square pixmap in the middle.
    For use as points cursor."""

    size = 25

    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)

    # Base measures
    width = 1
    center = 3  # Must be odd!
    rect_size = center + 2 * width
    square = rect_size + width * 4

    pen = QPen(Qt.white, 1)
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

    pen = QPen(Qt.black, 2)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.setPen(pen)

    # # Square
    painter.drawRect(
        (size - square) // 2 + 2,
        (size - square) // 2 + 2,
        square - 4,
        square - 4,
    )

    pen = QPen(Qt.black, 3)
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
        partial(remove_flash_animation, weakref.ref(widget))
    )

    widget._flash_animation.start()

    # now  set an actual time for the flashing and an intermediate color
    widget._flash_animation.setDuration(duration)
    widget._flash_animation.setKeyValueAt(0.1, QColor(*color))


def remove_flash_animation(widget_ref: weakref.ref[QWidget]):
    """Remove flash animation from widget.

    Parameters
    ----------
    widget_ref : QWidget
        Any Qt widget.
    """
    if widget_ref() is None:
        return
    widget = widget_ref()
    try:
        widget.setGraphicsEffect(None)
        del widget._flash_animation
    except RuntimeError:
        # RuntimeError: wrapped C/C++ object of type QtWidgetOverlay deleted
        pass


@contextmanager
def _maybe_allow_interrupt(qapp):
    """
    This manager allows to terminate a plot by sending a SIGINT. It is
    necessary because the running Qt backend prevents Python interpreter to
    run and process signals (i.e., to raise KeyboardInterrupt exception). To
    solve this one needs to somehow wake up the interpreter and make it close
    the plot window. We do this by using the signal.set_wakeup_fd() function
    which organizes a write of the signal number into a socketpair connected
    to the QSocketNotifier (since it is part of the Qt backend, it can react
    to that write event). Afterwards, the Qt handler empties the socketpair
    by a recv() command to re-arm it (we need this if a signal different from
    SIGINT was caught by set_wakeup_fd() and we shall continue waiting). If
    the SIGINT was caught indeed, after exiting the on_signal() function the
    interpreter reacts to the SIGINT according to the handle() function which
    had been set up by a signal.signal() call: it causes the qt_object to
    exit by calling its quit() method. Finally, we call the old SIGINT
    handler with the same arguments that were given to our custom handle()
    handler.
    We do this only if the old handler for SIGINT was not None, which means
    that a non-python handler was installed, i.e. in Julia, and not SIG_IGN
    which means we should ignore the interrupts.

    code from https://github.com/matplotlib/matplotlib/pull/13306
    """
    old_sigint_handler = signal.getsignal(signal.SIGINT)
    handler_args = None
    skip = False
    if old_sigint_handler in (None, signal.SIG_IGN, signal.SIG_DFL):
        skip = True
    else:
        wsock, rsock = socket.socketpair()
        wsock.setblocking(False)
        old_wakeup_fd = signal.set_wakeup_fd(wsock.fileno())
        sn = QSocketNotifier(rsock.fileno(), QSocketNotifier.Type.Read)

        # Clear the socket to re-arm the notifier.
        sn.activated.connect(lambda *args: rsock.recv(1))

        def handle(*args):
            nonlocal handler_args
            handler_args = args
            qapp.exit()

        signal.signal(signal.SIGINT, handle)
    try:
        yield
    finally:
        if not skip:
            wsock.close()
            rsock.close()
            sn.setEnabled(False)
            signal.set_wakeup_fd(old_wakeup_fd)
            signal.signal(signal.SIGINT, old_sigint_handler)
            if handler_args is not None:
                old_sigint_handler(*handler_args)


class Sentry(QObject):
    """Small object to trigger events across threads."""

    alerted = Signal()

    def alert(self, *_):
        self.alerted.emit()


def qt_might_be_rich_text(text) -> bool:
    """
    Check if a text might be rich text in a cross-binding compatible way.
    """
    if qtpy.PYSIDE2:
        from qtpy.QtGui import Qt as _Qt
    else:
        from qtpy.QtCore import Qt as _Qt

    try:
        return _Qt.mightBeRichText(text)
    except Exception:
        return bool(RICH_TEXT_PATTERN.search(text))
