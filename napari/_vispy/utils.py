from ..layers import Image, Points, Shapes, Surface, Tracks, Vectors
from ..utils import config
from .vispy_image_layer import VispyImageLayer
from .vispy_points_layer import VispyPointsLayer
from .vispy_shapes_layer import VispyShapesLayer
from .vispy_surface_layer import VispySurfaceLayer
from .vispy_tracks_layer import VispyTracksLayer
from .vispy_vectors_layer import VispyVectorsLayer

# Regular layers: no camera.
layer_to_visual = {
    Image: VispyImageLayer,
    Points: VispyPointsLayer,
    Shapes: VispyShapesLayer,
    Surface: VispySurfaceLayer,
    Vectors: VispyVectorsLayer,
    Tracks: VispyTracksLayer,
}

# Camera-dependent layers.
layer_to_visual_camera = {}

if config.async_loading:
    from ..layers.image.experimental.octree_image import OctreeImage
    from .experimental.vispy_tiled_image_layer import VispyTiledImageLayer

    layer_to_visual_camera = {OctreeImage: VispyTiledImageLayer}


def create_vispy_visual(layer, camera):
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
    # Check camera layers first.
    for layer_type, visual in layer_to_visual_camera.items():
        if isinstance(layer, layer_type):
            return visual(layer, camera)

    # Then regular layers.
    for layer_type, visual in layer_to_visual.items():
        if isinstance(layer, layer_type):
            return visual(layer)

    raise TypeError(
        f'Could not find VispyLayer for layer of type {type(layer)}'
    )
