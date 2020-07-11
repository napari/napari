from ...layers import Image, Labels, Points, Shapes, Surface, Vectors
from .qt_image_controls import QtImageControls
from .qt_points_controls import QtPointsControls
from .qt_shapes_controls import QtShapesControls
from .qt_labels_controls import QtLabelsControls
from .qt_surface_controls import QtSurfaceControls
from .qt_vectors_controls import QtVectorsControls


layer_to_controls = {
    Image: QtImageControls,
    Labels: QtLabelsControls,
    Points: QtPointsControls,
    Shapes: QtShapesControls,
    Surface: QtSurfaceControls,
    Vectors: QtVectorsControls,
}


def create_qt_layer_controls(layer):
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
