from ._base_layer import QtLayerProperties, QtLayerControls
from ._image_layer import QtImageProperties, QtImageControls
from ._markers_layer import QtMarkersProperties, QtMarkersControls
from ._vectors_layer import QtVectorsProperties
from ._shapes_layer import QtShapesProperties, QtShapesControls
from ._labels_layer import QtLabelsProperties, QtLabelsControls
from ._pyramid_layer import QtPyramidProperties, QtPyramidControls


def get_qt_properties(layer):
    name = 'Qt' + type(layer).__name__ + 'Properties'
    if name in locals():
        properties = exec(name + '(layer)')
    else:
        properties = QtLayerProperties(layer)

    return properties


def get_qt_controls(layer):
    name = 'Qt' + type(layer).__name__ + 'Controls'
    if name in locals():
        controls = exec(name + '(layer)')
    else:
        controls = QtLayerControls(layer)

    return controls
