import os

from ..layers import Image, Points, Shapes, Surface, Tracks, Vectors
from .vispy_image_layer import VispyImageLayer
from .vispy_points_layer import VispyPointsLayer
from .vispy_shapes_layer import VispyShapesLayer
from .vispy_surface_layer import VispySurfaceLayer
from .vispy_tracks_layer import VispyTracksLayer
from .vispy_vectors_layer import VispyVectorsLayer

_use_async = os.getenv("NAPARI_ASYNC", "0") != "0"

layer_to_visual = {
    Image: VispyImageLayer,
    Points: VispyPointsLayer,
    Shapes: VispyShapesLayer,
    Surface: VispySurfaceLayer,
    Vectors: VispyVectorsLayer,
    Tracks: VispyTracksLayer,
}

# Added experimental layer only when using async.
if _use_async:
    from ..layers.image.experimental import OctreeImage
    from .experimental.vispy_octree_image_layer import VispyOctreeImageLayer

    layer_to_visual[OctreeImage] = VispyOctreeImageLayer


def create_vispy_visual(layer):
    """Create vispy visual for a layer based on its layer type.

    Parameters
    ----------
    layer : napari.layers._base_layer.Layer
        Layer that needs its property widget created.

    Returns
    -------
    visual : vispy.scene.visuals.VisualNode
        Vispy visual node
    """
    for layer_type, visual in layer_to_visual.items():
        if isinstance(layer, layer_type):
            return visual(layer)

    raise TypeError(
        f'Could not find VispyLayer for layer of type {type(layer)}'
    )
