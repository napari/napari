from ..layers import Image, Labels, Points, Pyramid, Shapes, Vectors, Volume
from .vispy_image_layer import VispyImageLayer
from .vispy_labels_layer import VispyLabelsLayer
from .vispy_points_layer import VispyPointsLayer
from .vispy_pyramid_layer import VispyPyramidLayer
from .vispy_shapes_layer import VispyShapesLayer
from .vispy_vectors_layer import VispyVectorsLayer
from .vispy_volume_layer import VispyVolumeLayer


layer_to_visual = {
    Image: VispyImageLayer,
    Labels: VispyLabelsLayer,
    Points: VispyPointsLayer,
    Pyramid: VispyPyramidLayer,
    Shapes: VispyShapesLayer,
    Vectors: VispyVectorsLayer,
    Volume: VispyVolumeLayer,
}


def create_vispy_visual(layer):
    """
    Create vispy visual for a layer based on its layer type.

    Parameters
    ----------
        layer : napari.layers._base_layer.Layer
            Layer that needs its propetry widget created.

    Returns
    ----------
        visual : vispy.scene.visuals.VisualNode
            Vispy visual node
    """
    visual = layer_to_visual[type(layer)](layer)

    return visual
