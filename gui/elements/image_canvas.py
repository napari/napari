from OpenGL import GL
from OpenGL.raw.GL.VERSION.GL_1_0 import glMatrixMode, glLoadIdentity
from OpenGL.raw.GL.VERSION.GL_1_1 import GL_PROJECTION
from PyQt5 import QtWidgets, QtCore, QtGui, QtOpenGL
from OpenGL.GL.ARB.texture_rg import GL_R32F

import numpy as np
import ctypes
import time

from vispy import scene
from vispy.app import Canvas
from vispy.io import read_png
from vispy.scene import SceneCanvas
from vispy.util import load_data_file

from .panzoom import PanZoomCamera
from ..visuals.napari_image import NapariImage


class ImageCanvas(SceneCanvas):
    # get available interpolation methods
    interpolation_method_names = scene.visuals.Image(None).interpolation_functions
    interpolation_method_names = list(interpolation_method_names)
    interpolation_method_names.sort()
    interpolation_method_names.remove('sinc')  # does not work well on my machine

    # print(interpolation_method_names)


    def __init__(self, parent_widget, window_width, window_height):
        super(ImageCanvas, self).__init__(keys=None, vsync=True)

        self.size = window_width, window_height

        self.unfreeze()

        self.parent_widget = parent_widget
        # Set up a viewbox to display the image with interactive pan/zoom
        self.view = self.central_widget.add_view()
        self.image_visual = NapariImage(None,  parent=self.view.scene, method='auto')

        self.image = None
        self.brightness = 1
        self.interpolation_index = 0

        self.freeze()

        self.set_interpolation('nearest')

    def set_image(self, image, dimx=0, dimy=1):
        self.image = image

        self.image_visual.set_data(image)
        self.view.camera.set_range()

        # Set 2D camera (the camera will scale to the contents in the scene)
        self.view.camera = PanZoomCamera(aspect=1)
        # flip y-axis to have correct aligment
        self.view.camera.flip = (0, 1, 0)
        self.view.camera.set_range()
        # view.camera.zoom(0.1, (250, 200))

    def get_interpolation_name(self):
        return type(self).interpolation_method_names[self.interpolation_index]

    def set_interpolation(self, interpolation):
        self.set_interpolation_index(type(self).interpolation_method_names.index(interpolation))

    def increment_interpolation_index(self):
        self.set_interpolation_index(self.interpolation_index + 1)

    def set_interpolation_index(self, interpolation_index):
        self.interpolation_index = interpolation_index % len(type(self).interpolation_method_names)
        self.image_visual.interpolation = type(self).interpolation_method_names[self.interpolation_index]
        # print(self.image_visual.interpolation)
        self.update()

    def setBrightness(self, brightness):
        print("brightess = %f" % brightness)
        self.brightness = brightness
        self.update()

    def set_cmap(self, cmap):
        self.image_visual.cmap = cmap
        self.update()

    def on_key_press(self, event):
        # print("Sending to QT parent: %s " % event.key)
        self.parent_widget.on_key_press(event)
