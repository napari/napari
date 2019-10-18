from ...layers import Image, Labels, Points, Shapes, Surface, Text, Vectors
from .qt_base_layer import QtLayerControls
from .qt_image_base_layer import QtBaseImageControls
from .qt_image_layer import QtImageControls
from .qt_points_layer import QtPointsControls
from .qt_shapes_layer import QtShapesControls
from .qt_labels_layer import QtLabelsControls
from .qt_surface_layer import QtSurfaceControls
from .qt_text_layer import QtTextControls
from .qt_vectors_layer import QtVectorsControls


layer_to_controls = {
    Image: QtImageControls,
    Labels: QtLabelsControls,
    Points: QtPointsControls,
    Shapes: QtShapesControls,
    Surface: QtSurfaceControls,
    Text: QtTextControls,
    Vectors: QtVectorsControls,
}


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
