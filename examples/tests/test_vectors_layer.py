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

from skimage import data
import numpy as np


class TestVectorsLayer(unittest.TestCase):
    
    def setup(self):
        # create UI elements
        self.application = QApplication(sys.argv)

        # create Viewer, Windows
        self.view = Viewer()
        self.win = Window(Viewer(), show=False)
        self.viewer = self.win.viewer
        print('creating test UI vectors')
        return None

    #test parameters, bounds and exception handling on parameters
    def test_width(self):
        self.setup()
        layer = self.viewer.add_image(data.astronaut(), {})


        # test UI entry works
        # test programmatic entry works
        # test ipython entry works
        # test invalid value type (int vs float etc..) and handling
        return None
    
    def test_color(self):
        #test that incorrect colors are handled
        return None
    
    def test_vectorData_2d(self):
        self.setup()
        # pos1 = np.zeros(shape=(10,10,2), dtype=np.float32)
        pos = np.zeros(shape = (10, 2), dtype=np.float32)
        self.viewer.add_vectors(pos)
        with self.assertRaises(NotImplementedError):
            self.viewer.add_vectors(pos)
        # test with good sample data
        # test with bad sample data (error is handled)
        return None
    
    def test_vectorData_imageLike(self):
        self.setup()
        imvect = np.empty(shape=(10,10,2), dtype=np.float32)
        space = np.linspace(0, 9, 10)
        xv, yv = np.meshgrid(space, space)
        imvect[:,:, 0] = xv
        imvect[:,:, 1] = yv
        self.viewer.add_vectors(imvect)
        return None
    
    def test_connector_types(self):
        return None
    
    def test_averaging_callback(self):
        # test UI entry works
        # and programmatic entry
        # test ipython entry works
        # test invalid entries and handling (strings, tuples, floats, etc..)
        return None

    def test_length(self):
        # test UI entry works
        # test programmatic entry works
        # test ipython entry works
        # test range (min max)
        # test data type and handling (must be int or float)
        return None
    
    #test UI
    def test_update(self):
        return None
    
    def test_refresh(self):
        return None
    
    def test_get_shape(self):
        return None
    
    #can consider testing of slices across some vectors
    def test_slices(self):
        return None