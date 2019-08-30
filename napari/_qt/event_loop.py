import os
import sys
from contextlib import contextmanager

from qtpy.QtWidgets import QApplication


@contextmanager
def gui_qt():
    """Start a Qt event loop in which to run the application.

    Notes
    -----
    This context manager is not needed if running napari within an interactive
    IPython session. In this case, use the %gui=qt magic command, or start
    IPython with the Qt GUI event loop enabled by default by using
    ``ipython --gui=qt``.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    yield
    app.exec_()
