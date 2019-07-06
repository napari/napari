import os
import sys
from contextlib import contextmanager

from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QTimer


@contextmanager
def gui_qt():
    """Start a Qt event loop in which to run the application.

    Notes
    -----
    This context manager is not needed if running napari within an interactive
    IPython session. In this case, use the %gui=qt magic command, or start
    IPython with the Qt GUI event loop enabled by default by using
    ``ipython --gui=qt``.

    If the `NAPARI_TEST` environment variable is set to anything but `0`,
    will automatically quit after 0.5 seconds.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    yield
    if os.environ.get('NAPARI_TEST', '0') != '0':
        # quit app after 0.5 seconds
        timer = QTimer()
        timer.setInterval(500)
        timer.timeout.connect(app.quit)
        timer.start()
    app.exec_()
