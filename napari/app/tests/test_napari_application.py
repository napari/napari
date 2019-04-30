import sys
from PyQt5.QtWidgets import QApplication

from napari.app.napari_application import NapariApplication


def test_application_type():
    app = NapariApplication(sys.argv)
    assert isinstance(app, QApplication)
