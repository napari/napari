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


# get available interpolation methods
interpolation_method_names = scene.visuals.Image(None).interpolation_functions
interpolation_method_names = list(interpolation_method_names)
interpolation_method_names.sort()
interpolation_method_names.remove('sinc')  # does not work well on my machine

# print(interpolation_method_names)
index_to_name = interpolation_method_names.__getitem__
name_to_index = interpolation_method_names.index


class ImageCanvas(SceneCanvas):
    """Canvas to draw images on.

    Parameters
    ----------
    parent_widget : QWidget
        Parent widget.
    window_width : int
        Width of the window.
    window_height : int
        Height of the window.
    """
    def __init__(self, parent_widget, window_width, window_height):
        super().__init__(keys=None, vsync=True)

        self.size = window_width, window_height

        self.unfreeze()

        self.parent_widget = parent_widget
        # Set up a viewbox to display the image with interactive pan/zoom
        self.view = self.central_widget.add_view()

        # Set 2D camera (the camera will scale to the contents in the scene)
        self.view.camera = PanZoomCamera(aspect=1)
        # flip y-axis to have correct aligment
        self.view.camera.flip = (0, 1, 0)
        self.view.camera.set_range()
        # view.camera.zoom(0.1, (250, 200))

        self.image_visual = NapariImage(None,  parent=self.view.scene, method='auto')

        self.image = None
        self._brightness = 1
        self._interpolation_index = 0

        self.freeze()

        self.interpolation = 'nearest'

    def set_image(self, image, dimx=0, dimy=1):
        """Sets the image given the data.

        Parameters
        ----------
        image : array
            Image data to update with.
        dimx : int, optional
            Ordinal axis considered as the x-axis.
        dimy : int, optional
            Ordinal axis considered as the y-axis.
        """
        # TODO: use dimx, dimy for something
        self.image = image

        self.image_visual.set_data(image)
        self.view.camera.set_range()

    @property
    def interpolation(self):
        """string: Equipped interpolation method's name.
        """
        return index_to_name(self.interpolation_index)

    @interpolation.setter
    def interpolation(self, interpolation):
        self.interpolation_index = name_to_index(interpolation)

    @property
    def interpolation_index(self):
        """int: Index of the current interpolation method equipped.
        """
        return self._interpolation_index

    @interpolation_index.setter
    def interpolation_index(self, interpolation_index):
        intp_index = interpolation_index % len(interpolation_method_names)
        self._interpolation_index = intp_index
        self.image_visual.interpolation = index_to_name(intp_index)
        # print(self.image_visual.interpolation)
        self.update()

    @property
    def brightness(self):
        """float: Image brightness.
        """
        return self._brightness

    @brightness.setter
    def brightness(self, brightness):
        # TODO: actually implement this
        print("brightess = %f" % brightness)
        if not 0.0 < brightness < 1.0:
            raise ValueError('brightness must be between 0-1, not '
                             + brightness)

        self.brightness = brightness
        self.update()

    @property
    def cmap(self):
        """string: Color map.
        """
        return self.image_visual.cmap

    @cmap.setter
    def cmap(self, cmap):
        self.image_visual.cmap = cmap
        self.update()

    def on_key_press(self, event):
        # print("Sending to QT parent: %s " % event.key)
        self.parent_widget.on_key_press(event)
