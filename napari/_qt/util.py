import numpy as np
from qtpy import API_NAME
from qtpy.QtCore import QObject, QThread


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
    Worker: type(QObject), *args, start=False, connections=None, **kwargs
):
    """This is a convenience method to start a worker in a Qthread

    It follows the pattern described here:
    https://www.qt.io/blog/2010/06/17/youre-doing-it-wrong

    all *args, **kwargs will be passed to the worker class on instantiation.


    Parameters
    ----------
    Worker : QObject
        QObject type that implements a work() method.  The Worker should also
        emit a finished signal when the work is done.
    start : bool
        If True, worker will be started immediately, otherwise, you must
        manually start the worker.
    connections: dict, optional
        Optional dictionary of {signal: function} to connect to the new worker.
        for instance:  connections = {'incremented': myfunc} will result in:
        worker.incremented.connect(myfunc)
    """

    if not isinstance(connections, (dict, type(None))):
        raise TypeError('connections parameter must be a dict')

    thread = QThread()
    worker = Worker(*args, **kwargs)
    worker.moveToThread(thread)
    # not sure why connect(worker.work) isn't working!
    # TODO: figure out why the bare functions aren't connecting
    thread.started.connect(lambda: worker.work())
    worker.finished.connect(lambda: thread.quit())
    worker.finished.connect(lambda: worker.deleteLater())
    thread.finished.connect(lambda: thread.deleteLater())

    if connections:
        [getattr(worker, key).connect(val) for key, val in connections.items()]

    if start:
        thread.start()  # usually need to connect stuff before starting
    return worker, thread
