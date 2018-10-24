import unittest
import sys
from PyQt5.QtWidgets import QApplication

from ..gui.napari_application import NapariApplication


class NapariApplicationTest(unittest.TestCase):

    def test_contructor_type(self):
        app = NapariApplication(sys.argv)
        self.assertIsInstance(app, QApplication, msg=None)

