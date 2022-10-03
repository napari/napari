from typing import Tuple

import numpy as np
from vispy.scene.widgets.viewbox import ViewBox

from ...layers import (
    Image,
    Labels,
    Layer,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)
from ...utils.config import async_octree
from ...utils.translations import trans
from ..layers.base import VispyBaseLayer
from ..layers.image import VispyImageLayer
from ..layers.labels import VispyLabelsLayer
from ..layers.points import VispyPointsLayer
from ..layers.shapes import VispyShapesLayer
from ..layers.surface import VispySurfaceLayer
from ..layers.tracks import VispyTracksLayer
from ..layers.vectors import VispyVectorsLayer

layer_to_visual = {
    Image: VispyImageLayer,
    Labels: VispyLabelsLayer,
    Points: VispyPointsLayer,
    Shapes: VispyShapesLayer,
    Surface: VispySurfaceLayer,
    Vectors: VispyVectorsLayer,
    Tracks: VispyTracksLayer,
}


if async_octree:
    from ...layers.image.experimental.octree_image import _OctreeImageBase
    from ..experimental.vispy_tiled_image_layer import VispyTiledImageLayer

    # Insert _OctreeImageBase in front so it gets picked over plain Image.
    new_mapping = {_OctreeImageBase: VispyTiledImageLayer}
    new_mapping.update(layer_to_visual)
    layer_to_visual = new_mapping


def create_vispy_layer(layer: Layer) -> VispyBaseLayer:
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
    ndim: int,
    dims_displayed: Tuple[int],
) -> np.ndarray:
    """Calculate the unit vector pointing in the direction of the view.

    This is only for 3D viewing, so it returns None when
    len(dims_displayed) == 2.
    Adapted From:
    https://stackoverflow.com/questions/37877592/
        get-view-direction-relative-to-scene-in-vispy/37882984

    Parameters
    ----------
    view : vispy.scene.widgets.viewbox.ViewBox
        The vispy view box object to get the view direction from.
    ndim : int
        The number of dimensions in the full nD dims model.
        This is typically from viewer.dims.ndim
    dims_displayed : Tuple[int]
        The indices of the dims displayed in the viewer.
        This is typically from viewer.dims.displayed.

    Returns
    -------
    view_vector : np.ndarray
        Unit vector in the direction of the view in scene coordinates.
        Axes are ordered zyx. If the viewer is in 2D
        (i.e., len(dims_displayed) == 2), view_vector is None.
    """
    # only return a vector when viewing in 3D
    if len(dims_displayed) == 2:
        return None

    tform = view.scene.transform
    w, h = view.canvas.size

    # get a point at the center of the canvas
    # (homogeneous screen coords)
    screen_center = np.array([w / 2, h / 2, 0, 1])

    # find a point just in front of the center point
    # transform both to world coords and find the vector
    d1 = np.array([0, 0, 1, 0])
    point_in_front_of_screen_center = screen_center + d1
    p1 = tform.imap(point_in_front_of_screen_center)
    p0 = tform.imap(screen_center)
    d2 = p1 - p0

    # in 3D world coordinates
    d3 = d2[0:3]
    d4 = d3 / np.linalg.norm(d3)

    # data are ordered xyz on vispy Volume
    d4 = d4[[2, 1, 0]]
    view_dir_world = np.zeros((ndim,))
    for i, d in enumerate(dims_displayed):
        view_dir_world[d] = d4[i]

    return view_dir_world
