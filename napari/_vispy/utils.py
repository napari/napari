from typing import Dict

from ..layers import Image, Layer, Points, Shapes, Surface, Tracks, Vectors
from ..utils import config
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


def _get_octree_visual_class() -> VispyBaseLayer:
    """Return which OctreeImage layer visual class to create.

    OctreeImage layer supports two types of visuals:
    # 1) VispyCompoundImageLayer - separate ImageVisuals
    # 2) VispyTiledImageLayer - one TiledImageVisual

    Return
    ------
    VispyBaseLayer
        The visual layer class to create.
    """

    if config.create_image_type == config.CREATE_IMAGE_COMPOUND:
        from .experimental.vispy_compound_image_layer import (
            VispyCompoundImageLayer,
        )

        return VispyCompoundImageLayer
    else:
        from .experimental.vispy_tiled_image_layer import VispyTiledImageLayer

        return VispyTiledImageLayer


def get_layer_to_visual() -> Dict[Layer, VispyBaseLayer]:
    """Get the layer to visual mapping.

    We modify the layer layer to visual mapping for octree.

    Returns
    -------
    Dict[Layer, VispyBaseLayer]
        The mapping from layer to visual.
    """
    if not config.create_octree_image():
        return layer_to_visual  # The normal non-experimental version.
    else:
        # OctreeImage layer with one of two types of visuals.
        from ..layers.image.experimental.octree_image import OctreeImage

        # Insert OctreeImage in front so it gets picked over plain Image.
        new_mapping = {OctreeImage: _get_octree_visual_class()}
        new_mapping.update(layer_to_visual)
        return new_mapping


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
    for layer_type, visual_class in get_layer_to_visual().items():
        if isinstance(layer, layer_type):
            return visual_class(layer)

    raise TypeError(
        f'Could not find VispyLayer for layer of type {type(layer)}'
    )
