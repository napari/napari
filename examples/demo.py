import sys

import vispy
from PyQt5.QtWidgets import QApplication, QAction
import numpy as np

from napari_gui.elements import Window, Viewer

from vispy.visuals.transforms import STTransform

from skimage import data


def open_2Drgb(win):
    # opening a 2D RGB image:
    image = data.astronaut()
    meta = dict(name='BlueMarble', itype='rgb')
    win.viewer.add_image(image, meta)
    win.show()


def open_2Dsc(win):
    # opening a 2D single channel image:
    h = 5120
    w = 5120
    Y, X = np.ogrid[-2.5:2.5:h * 1j, -2.5:2.5:w * 1j]
    image = np.empty((h, w), dtype=np.float32)
    image[:] = np.random.rand(h, w)
    image[-30:] = np.linspace(0, 1, w)
    meta = dict(name='2D1C', itype='mono')
    viewer = win.viewer
    layer = viewer.add_image(image, meta)
    layer.cmap = 'viridis'
    win.show()


def open_3Dsc(win):
    # opening a 3D single channel image:
    h = 512
    w = 512
    d = 512
    Z, Y, X = np.ogrid[-2.5:2.5:h * 1j, -2.5:2.5:w * 1j, -2.5:2.5:d * 1j]
    image = np.empty((h, w, d), dtype=np.float32)
    image[:] = np.exp(- X ** 2 - Y ** 2 - Z ** 2)  # * (1. + .5*(np.random.rand(h, w)-.5))
    # image[-30:] = np.linspace(0, 1, w)
    meta = dict(name='3D1C', itype='mono')
    viewer = win.viewer
    layer = viewer.add_image(image, meta)
    layer.cmap = 'blues'
    win.show()


def open_4Dsc(win):
    # opening a 4D single channel image:
    h = 32
    w = 32
    d = 64
    b = 64
    C, Z, Y, X = np.ogrid[-2.5:2.5:h * 1j, -2.5:2.5:w * 1j, -2.5:2.5:d * 1j, -2.5:2.5:b * 1j]
    image = np.empty((h, w, d, b), dtype=np.float32)
    image[:] = np.exp(- X ** 2 - Y ** 2 - Z ** 2 - C ** 2)  # * (1. + .5*(np.random.rand(h, w)-.5))
    # image[-30:] = np.linspace(0, 1, w)
    meta = dict(name='4D1C', itype='mono')
    viewer = win.viewer
    layer = viewer.add_image(image, meta)
    layer.cmap = 'blues'
    layer.interpolation = 'spline36'
    win.show()


def open_multi(win):
    # opening a 3D and 4D single-channel images in the same viewer
    viewer = win.viewer

    h = 64
    w = 64
    d = 64
    Z, Y, X = np.ogrid[-2.5:2.5:h * 1j, -2.5:2.5:w * 1j, -2.5:2.5:d * 1j]
    image3D = np.empty((h, w, d), dtype=np.float32)
    image3D[:] = np.exp(- X ** 2 - Y ** 2 - Z ** 2)  # * (1. + .5*(np.random.rand(h, w)-.5))
    # image[-30:] = np.linspace(0, 1, w)
    meta = dict(name='3D1C', itype='mono')
    layer = viewer.add_image(image3D, meta)
    layer.cmap = 'blues'
    layer.interpolation = 'spline36'

    h = 64
    w = 64
    d = 64
    b = 64
    C, Z, Y, X = np.ogrid[-2.5:2.5:h * 1j, -2.5:2.5:w * 1j, -2.5:2.5:d * 1j, -2.5:2.5:b * 1j]
    image4D = np.empty((h, w, d, b), dtype=np.float32)
    image4D[:] = np.exp(- X ** 2 - Y ** 2 - Z ** 2 - C ** 2)  # * (1. + .5*(np.random.rand(h, w)-.5))
    # image[-30:] = np.linspace(0, 1, w)
    meta = dict(name='4D1C', itype='mono')
    layer = viewer.add_image(image4D, meta)
    layer.cmap = 'blues'
    layer.interpolation = 'spline36'

    scale = image3D.shape[1] / image4D.shape[1]

    layer.translate = [image3D.shape[1]]
    layer.scale = [scale] * 2

    viewer.camera.set_range((0, w * 2), (0, h))
    win.show()

if __name__ == '__main__':
    # starting
    application = QApplication(sys.argv)

    #open_2Drgb(Window(Viewer(), show=False))
    #open_2Dsc(Window(Viewer(), show=False))
    open_3Dsc(Window(Viewer(), show=False))
    #open_4Dsc(Window(Viewer(), show=False))
    #open_multi(Window(Viewer(), show=False))

    sys.exit(application.exec_())
