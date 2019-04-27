import sys
from qtpy.QtWidgets import QApplication

from napari_application import NapariApplication


def test_application_type():
    app = NapariApplication(sys.argv)
    assert isinstance(app, QApplication)
