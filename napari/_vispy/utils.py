from ..layers import Image, Layer, Points, Shapes, Surface, Tracks, Vectors
from ..utils.config import async_octree
from .vispy_base_layer import VispyBaseLayer
from .vispy_image_layer import VispyImageLayer
from .vispy_points_layer import VispyPointsLayer
from .vispy_shapes_layer import VispyShapesLayer
from .vispy_surface_layer import VispySurfaceLayer
from .vispy_tracks_layer import VispyTracksLayer
from .vispy_vectors_layer import VispyVectorsLayer

layer_to_visual = {
    Image: VispyImageLayer,
    Points: VispyPointsLayer,
    Shapes: VispyShapesLayer,
    Surface: VispySurfaceLayer,
    Vectors: VispyVectorsLayer,
    Tracks: VispyTracksLayer,
}


if async_octree:
    from ..layers.image.experimental.octree_image import OctreeImage
    from .experimental.vispy_tiled_image_layer import VispyTiledImageLayer

    # Insert OctreeImage in front so it gets picked over plain Image.
    new_mapping = {OctreeImage: VispyTiledImageLayer}
    new_mapping.update(layer_to_visual)
    layer_to_visual = new_mapping


def create_vispy_visual(layer: Layer) -> VispyBaseLayer:
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
    for layer_type, visual_class in layer_to_visual.items():
        if isinstance(layer, layer_type):
            return visual_class(layer)

    raise TypeError(
        f'Could not find VispyLayer for layer of type {type(layer)}'
    )
