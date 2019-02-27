#!/usr/bin/env python
# title           : test_vectors_layer.py
# description     :This will create a header for a python script.
# author          :bryant.chhun
# date            :1/25/19
# version         :0.0
# usage           :python this_python_file.py -flags
# notes           :
# python_version  :3.6

import unittest
import sys

from PyQt5.QtWidgets import QApplication
from napari_gui import Viewer, Window
from napari_gui.layers._vectors_layer.model import InvalidDataFormatError

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

        N=10
        M=5
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

        N=10
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
        N=10
        M=5
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
        # self.a_create_env()
        N=10
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
        # self.a_create_env()
        N=10
        pos = np.zeros(shape=(N, 4), dtype=np.float32)
        pos2 = np.zeros(shape=(N, N, 2), dtype=np.float32)
        try:
            layer = self.viewer.add_vectors(pos)
            layer.vectors = pos2
        except Exception as ex:
            self.fail("exception thrown when : "+str(ex))
        return None
    
    def test_averaging_callback(self):
        """
        test assignment of function to averaging callback
        :return:
        """
        def testfunc(input):
            return input

        # self.a_create_env()
        pos = np.zeros(shape=(10, 10, 2), dtype=np.float32)
        try:
            layer = self.viewer.add_vectors(pos)
            layer.averaging_bind_to(testfunc)
            layer.averaging_bind_to(testfunc)
            layer.averaging_bind_to(testfunc)
            self.assertEqual(len(layer._avg_observers), 3)
        except Exception as ex:
            self.fail("exception thrown when  "+str(ex))
        return None

    def test_length_callback(self):
        """
        test assignment of function to length callback
        :return:
        """
        def testfunc(input):
            return input

        # self.a_create_env()
        pos = np.zeros(shape=(10, 10, 2), dtype=np.float32)
        try:
            layer = self.viewer.add_vectors(pos)
            layer.length_bind_to(testfunc)
            layer.length_bind_to(testfunc)
            layer.length_bind_to(testfunc)
            self.assertEqual(len(layer._len_observers), 3)
        except Exception as ex:
            self.fail("exception thrown when : "+str(ex))
        return None

    def test_callback(self):
        """
        test ability to change underlying data upon callback response
        :return:
        """
        def testfunc(input):
            layer.vectors = np.zeros(shape=(5,5,2), dtype=np.uint8)
            return input

        # self.a_create_env()
        pos = np.zeros(shape=(10, 10, 2), dtype=np.float32)
        try:
            layer = self.viewer.add_vectors(pos)
            self.assertEqual(layer._current_shape, (10,10,2))

            layer.length_bind_to(testfunc)
            layer.length = 1

            self.assertEqual(layer._current_shape, (5,5,2))
        except Exception as ex:
            self.fail("exception thrown when : "+str(ex))
        return None