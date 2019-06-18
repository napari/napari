from ._base_layer import QtLayerProperties, QtLayerControls
from ._image_layer import QtImageProperties, QtImageControls
from ._markers_layer import QtMarkersProperties, QtMarkersControls
from ._vectors_layer import QtVectorsProperties
from ._shapes_layer import QtShapesProperties, QtShapesControls
from ._labels_layer import QtLabelsProperties, QtLabelsControls
from ._pyramid_layer import QtPyramidProperties, QtPyramidControls


def create_qt_properties(layer):
    """
    Create a qt properties widget for a layer based on its layer type.

    Parameters
    ----------
        layer : napari.layers._base_layer.Layer
            Layer that needs its propetry widget created.

    Returns
    ----------
        properties : napari.layers._base_layer.QtLayerProperties
            Qt propetry widget
    """
    name = 'Qt' + type(layer).__name__ + 'Properties'
    try:
        qt_props = globals()[name]
        properties = qt_props(layer)
    except KeyError:
        properties = QtLayerProperties(layer)

    return properties


def create_qt_controls(layer):
    """
    Create a qt controls widget for a layer based on its layer type.

    Parameters
    ----------
        layer : napari.layers._base_layer.Layer
            Layer that needs its controls widget created.

    Returns
    ----------
        controls : napari.layers._base_layer.QtLayerControls
            Qt controls widget
    """
    name = 'Qt' + type(layer).__name__ + 'Controls'
    try:
        qt_controls = globals()[name]
        controls = qt_controls(layer)
    except KeyError:
        controls = QtLayerControls(layer)

    return controls
