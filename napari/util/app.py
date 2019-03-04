import sys
from contextlib import contextmanager
from PyQt5.QtWidgets import QApplication


@contextmanager
def app_context():
    app = QApplication.instance() or QApplication(sys.argv)
    yield
    app.exec()
