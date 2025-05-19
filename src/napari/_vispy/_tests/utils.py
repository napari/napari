import numpy as np
from vispy.visuals import VolumeVisual
from vispy.visuals.transforms.linear import STTransform

from napari._vispy.layers.image import VispyImageLayer


def vispy_image_scene_size(vispy_image: VispyImageLayer) -> np.ndarray:
    """Calculates the size of a vispy image/volume in 3D space.

    The size is the shape of the node's data multiplied by the
    node's transform scale factors.

    Returns
    -------
    np.ndarray
        The size of the node as a 3-vector of the form (x, y, z).
    """
    node = vispy_image.node
    data = node._last_data if isinstance(node, VolumeVisual) else node._data
    # Only use scale to ignore translate offset used to center top-left pixel.
    transform = STTransform(scale=np.diag(node.transform.matrix))
    # Vispy uses an xy-style ordering, whereas numpy uses a rc-style
    # ordering, so reverse the shape before applying the transform.
    size = transform.map(data.shape[::-1])
    # The last element should always be one, so ignore it.
    return size[:3]
