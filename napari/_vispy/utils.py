import numpy as np
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


def quaternion2euler(quaternion, degrees=False):
    """Converts a quaternion into an euler angles representation.

    Parameters
    ----------
    quaternion : vispy.util.Quaternion
        Quaternion for conversion.
    degrees : bool
        If output is returned in degrees or radians.

    Returns
    -------
    angles : 3-tuple
        Euler angles in (rx, ry, rz) order.
    """
    q = quaternion
    angles = (
        np.arctan2(
            2 * (q.w * q.z + q.y * q.x), 1 - 2 * (q.y * q.y + q.z * q.z),
        ),
        np.arcsin(2 * (q.w * q.y - q.z * q.x)),
        np.arctan2(
            2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y),
        ),
    )
    if degrees:
        return tuple(np.degrees(angles))
    else:
        return angles
