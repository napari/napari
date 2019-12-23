from ..layers import Image, Labels, Points, Shapes, Surface, Vectors
from .vispy_image_layer import VispyImageLayer
from .vispy_points_layer import VispyPointsLayer
from .vispy_shapes_layer import VispyShapesLayer
from .vispy_vectors_layer import VispyVectorsLayer
from .vispy_surface_layer import VispySurfaceLayer


layer_to_visual = {
    Image: VispyImageLayer,
    Labels: VispyImageLayer,
    Points: VispyPointsLayer,
    Shapes: VispyShapesLayer,
    Surface: VispySurfaceLayer,
    Vectors: VispyVectorsLayer,
}


def create_vispy_visual(layer):
    """Create vispy visual for a layer based on its layer type.

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
