import numpy as np
from vispy.scene.widgets.viewbox import ViewBox

from ..layers import (
    Image,
    Labels,
    Layer,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)
from ..utils.config import async_octree
from ..utils.translations import trans
from .vispy_base_layer import VispyBaseLayer
from .vispy_image_layer import VispyImageLayer
from .vispy_points_layer import VispyPointsLayer
from .vispy_shapes_layer import VispyShapesLayer
from .vispy_surface_layer import VispySurfaceLayer
from .vispy_tracks_layer import VispyTracksLayer
from .vispy_vectors_layer import VispyVectorsLayer

layer_to_visual = {
    Image: VispyImageLayer,
    Labels: VispyImageLayer,
    Points: VispyPointsLayer,
    Shapes: VispyShapesLayer,
    Surface: VispySurfaceLayer,
    Vectors: VispyVectorsLayer,
    Tracks: VispyTracksLayer,
}


if async_octree:
    from ..layers.image.experimental.octree_image import _OctreeImageBase
    from .experimental.vispy_tiled_image_layer import VispyTiledImageLayer

    # Insert _OctreeImageBase in front so it gets picked over plain Image.
    new_mapping = {_OctreeImageBase: VispyTiledImageLayer}
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
        trans._(
            'Could not find VispyLayer for layer of type {dtype}',
            deferred=True,
            dtype=type(layer),
        )
    )


def get_view_direction_in_scene_coordinates(
    view: ViewBox,
    dims_point,
    dims_displayed,
) -> np.ndarray:
    """calculate the unit vector pointing in the direction of the view

    Adapted From:
    https://stackoverflow.com/questions/37877592/
        get-view-direction-relative-to-scene-in-vispy/37882984

    Parameters
    ----------
    view : vispy.scene.widgets.viewbox.ViewBox
        The vispy view box object to get the view direction from.

    Returns
    -------
    view_vector : np.ndarray
        Unit vector in the direction of the view in scene coordinates.
        Axes are ordered zyx.
    """
    tform = view.scene.transform
    w, h = view.canvas.size
    # in homogeneous screen coordinates
    screen_center = np.array([w / 2, h / 2, 0, 1])
    d1 = np.array([0, 0, 1, 0])
    point_in_front_of_screen_center = screen_center + d1
    p1 = tform.imap(point_in_front_of_screen_center)
    p0 = tform.imap(screen_center)
    assert abs(p1[3] - 1.0) < 1e-5
    assert abs(p0[3] - 1.0) < 1e-5
    d2 = p1 - p0
    assert abs(d2[3]) < 1e-5
    # in 3D screen coordinates
    d3 = d2[0:3]
    d4 = d3 / np.linalg.norm(d3)
    d4 = d4[[2, 1, 0]]
    view_dir_world = list(dims_point)
    for i, d in enumerate(dims_displayed):
        view_dir_world[d] = d4[i]

    return view_dir_world
