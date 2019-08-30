from ...layers import Image, Labels, Points, Pyramid, Shapes, Vectors
from .qt_base_layer import QtLayerProperties, QtLayerControls
from .qt_image_layer import QtImageProperties, QtImageControls
from .qt_points_layer import QtPointsProperties, QtPointsControls
from .qt_vectors_layer import QtVectorsProperties
from .qt_shapes_layer import QtShapesProperties, QtShapesControls
from .qt_labels_layer import QtLabelsProperties, QtLabelsControls


layer_to_properties = {
    Image: QtImageProperties,
    Labels: QtLabelsProperties,
    Points: QtPointsProperties,
    Pyramid: QtImageProperties,
    Shapes: QtShapesProperties,
    Vectors: QtVectorsProperties,
}


layer_to_controls = {
    Image: QtImageControls,
    Labels: QtLabelsControls,
    Points: QtPointsControls,
    Pyramid: QtImageControls,
    Shapes: QtShapesControls,
    Vectors: QtLayerControls,
}


def create_qt_properties(layer):
    """
    Create a qt properties widget for a layer based on its layer type.

    Parameters
    ----------
        layer : napari.layers._base_layer.Layer
            Layer that needs its propetry widget created.

    Returns
    ----------
        properties : napari.layers.base.QtLayerProperties
            Qt propetry widget
    """
    properties = layer_to_properties[type(layer)](layer)

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
        controls : napari.layers.base.QtLayerControls
            Qt controls widget
    """
    controls = layer_to_controls[type(layer)](layer)

    return controls
