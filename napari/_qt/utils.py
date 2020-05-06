import inspect
from contextlib import contextmanager
from functools import lru_cache, wraps
from typing import Type

import numpy as np
from qtpy import API_NAME
from qtpy.QtCore import QObject, QSize, Qt, QThread, Signal, Slot
from qtpy.QtGui import QCursor, QDrag, QImage, QPainter, QPixmap
from qtpy.QtWidgets import QGraphicsOpacityEffect, QListWidget


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
    if API_NAME == 'PySide2':
        arr = np.array(b).reshape(h, w, c)
    else:
        b.setsize(h * w * c)
        arr = np.frombuffer(b, np.uint8).reshape(h, w, c)

    # Format of QImage is ARGB32_Premultiplied, but color channels are
    # reversed.
    arr = arr[:, :, [2, 1, 0, 3]]
    return arr


def new_worker_qthread(
    Worker: Type[QObject],
    *args,
    start_thread=False,
    connections=None,
    **kwargs,
):
    """This is a convenience method to start a worker in a Qthread

    It follows the pattern described here:
    https://www.qt.io/blog/2010/06/17/youre-doing-it-wrong
    and
    https://doc.qt.io/qt-5/qthread.html#details

    all *args, **kwargs will be passed to the Worker class on instantiation.

    Parameters
    ----------
    Worker : QObject
        QObject type that implements a work() method.  The Worker should also
        emit a finished signal when the work is done.
    start_thread : bool
        If True, thread will be started immediately, otherwise, thread must
        be manually started with thread.start().
    connections: dict, optional
        Optional dictionary of {signal: function} to connect to the new worker.
        for instance:  connections = {'incremented': myfunc} will result in:
        worker.incremented.connect(myfunc)

    Examples
    --------
    Create some QObject that has a long-running work method:

    >>> class Worker(QObject):
    ...
    ...     finished = Signal()
    ...     increment = Signal(int)
    ...
    ...     def __init__(self, argument):
    ...         super().__init__()
    ...         self.argument = argument
    ...
    ...     @Slot()
    ...     def work(self):
    ...         # some long running task...
    ...         import time
    ...         for i in range(10):
    ...             time.sleep(1)
    ...             self.increment.emit(i)
    ...         self.finished.emit()
    ...
    >>> worker, thread = new_worker_qthread(
    ...     Worker,
    ...     'argument',
    ...     start_thread=True,
    ...     connections={'increment': print},
    ... )



    >>> print([i for i in example_generator(4)])
    [0, 1, 2, 3]

    """

    if not isinstance(connections, (dict, type(None))):
        raise TypeError('connections parameter must be a dict')

    thread = QThread()
    worker = Worker(*args, **kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.work)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    if connections:
        [getattr(worker, key).connect(val) for key, val in connections.items()]

    if start_thread:
        thread.start()  # sometimes need to connect stuff before starting
    return worker, thread


@contextmanager
def qt_signals_blocked(obj):
    """Context manager to temporarily block signals from `obj`"""
    obj.blockSignals(True)
    yield
    obj.blockSignals(False)


def disable_with_opacity(obj, widget_list, disabled):
    """Set enabled state on a list of widgets. If disabled, decrease opacity"""
    for wdg in widget_list:
        widget = getattr(obj, wdg)
        widget.setEnabled(obj.layer.editable)
        op = QGraphicsOpacityEffect(obj)
        op.setOpacity(1 if obj.layer.editable else 0.5)
        widget.setGraphicsEffect(op)


@lru_cache(maxsize=64)
def square_pixmap(size):
    """Create a white/black hollow square pixmap. For use as labels cursor."""
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setPen(Qt.white)
    painter.drawRect(0, 0, size - 1, size - 1)
    painter.setPen(Qt.black)
    painter.drawRect(1, 1, size - 3, size - 3)
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

    Example
    -------
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


class GeneratorWorker(QObject):
    """QObject that wraps a long-running generator function.

    Parameters
    ----------
    func : callable
        The function being wrapped.  Must return a generator
    *args, **kwargs: passed to func

    """

    started = Signal()
    yielded = Signal(object)
    returned = Signal(object)  # perhaps combine with finished?
    errored = Signal(object)
    finished = Signal()

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.gen = func(*args, **kwargs)
        self._incoming = None
        self._abort = False
        self.init = next(self.gen)
        if isinstance(self.init, dict):
            self.__dict__.update(**self.init)

    @Slot()
    def work(self):
        self.started.emit()
        while True:
            if self._abort:
                self.returned.emit('Aborted')
                break
            try:
                self.yielded.emit(self.gen.send(self._next_value()))
            except StopIteration as exc:
                self.returned.emit(exc.value)
                break
            except Exception as exc:
                self.errored.emit(exc)
                break
        self.finished.emit()

    def send(self, value):
        self._incoming = value

    def abort(self):
        self._abort = True

    def _next_value(self):
        out = None
        if self._incoming is not None:
            out = self._incoming
            self._incoming = None
        return out


def as_generatorfunction(func):
    """Turns a regular function (single return) into a generator function."""

    @wraps(func)
    def genwrapper(*args, **kwargs):
        yield
        return func(*args, **kwargs)

    return genwrapper


def qthreaded(func):
    """Decorator that decorates a generator and puts it in a QThread.

    Parameters
    ----------
    func : callable
        Function that returns a generator

    Returns
    -------
    callable
        function that creates a worker, puts it in a new thread and returns
        a two-tuple (worker, thread)
    """

    if inspect.isgeneratorfunction(func):
        _func = func
    else:
        _func = as_generatorfunction(func)

    @wraps(_func)
    def wrapper(*args, **kwargs):
        return new_worker_qthread(GeneratorWorker, _func, *args, **kwargs)

    return wrapper
