"""VispyTiledImageLayer class.
"""

from ..vispy_image_layer import VispyImageLayer


class VispyTiledImageLayer(VispyImageLayer):
    """Tiled images using a single TiledImageVisual."""

    def __init__(self, layer):
        super().__init__(layer)
