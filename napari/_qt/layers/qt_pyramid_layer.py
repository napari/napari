from ...layers import Pyramid
from .qt_image_layer import QtImageControls, QtImageProperties


class QtPyramidControls(QtImageControls, layer=Pyramid):
    pass


class QtPyramidProperties(QtImageProperties, layer=Pyramid):
    pass
