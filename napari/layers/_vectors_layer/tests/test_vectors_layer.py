import unittest
import sys

from PyQt5.QtWidgets import QApplication
from napari import Viewer, Window
from ..model import InvalidDataFormatError

import numpy as np


class TestVectorsLayer(unittest.TestCase):
    """
    these tests are performed on the same Napari viewer instance.
    """

    application = QApplication(sys.argv)
    view = Viewer()
    win = Window(Viewer(), show=False)
    viewer = win.viewer

    def test_vectorData_datatype0(self):
        """
        test data of image-like shape
        :return:
        """

        N = 10
        M = 5
        pos = np.zeros(shape=(N, M, 2), dtype=np.float32)
        try:
            self.viewer.add_vectors(pos)
        except Exception as ex:
            self.fail("exception thrown when creating coord-like vector layer: "+str(ex))
        return None

    def test_vectorData_datatype1(self):
        """
        test data of coord-like shape
        :return:
        """

        N = 10
        pos = np.zeros(shape=(N, 4), dtype=np.float32)
        try:
            self.viewer.add_vectors(pos)
        except Exception as ex:
            self.fail("exception thrown when creating coord-like vector layer: "+str(ex))
        return None

    def test_vectorData_datatype_notimpl(self):
        """
        test data of improper shape
        :return:
        """
        N = 10
        M = 5
        pos = np.zeros(shape=(N, M, 4), dtype=np.float32)
        try:
            self.viewer.add_vectors(pos)
        except InvalidDataFormatError:
            self.assertRaises(InvalidDataFormatError)
        except Exception as ex:
            self.fail("exception thrown when creating not implemented vector layer: "+str(ex))
        return None

    def test_vectorData_datatype_notimpl2(self):
        """
        test data of vispy-coordinate shape (not allowed)
        :return:
        """
        N = 10
        pos = np.zeros(shape=(N, 2), dtype=np.float32)
        try:
            self.viewer.add_vectors(pos)
        except InvalidDataFormatError:
            self.assertRaises(InvalidDataFormatError)
        except Exception as ex:
            self.fail("exception thrown when creating not implemented vector layer: "+str(ex))
        return None
    
    def test_vectorData_image_assignment(self):
        """
        test replacing vector data after layer construction
        :return:
        """
        N = 10
        pos = np.zeros(shape=(N, 4), dtype=np.float32)
        pos2 = np.zeros(shape=(N, N, 2), dtype=np.float32)
        try:
            layer = self.viewer.add_vectors(pos)
            layer.vectors = pos2
        except Exception as ex:
            self.fail("exception thrown when : "+str(ex))
        return None
