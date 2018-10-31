import pytest
import sys
from PyQt5.QtWidgets import QApplication

from napari_application import NapariApplication


class NapariApplicationTest(object):

    def test_contructor_type(self):
        app = NapariApplication(sys.argv)
        assert isinstance(app, QApplication)

