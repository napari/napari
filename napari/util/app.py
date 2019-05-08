import os
import sys
from contextlib import contextmanager

from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QTimer


@contextmanager
def app_context():
    """Generate app context.

    Notes
    -----
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
