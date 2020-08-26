"""LayerInfo class.
"""
import logging
import weakref

LOGGER = logging.getLogger("napari.async")


class LayerInfo:
    """Information about one layer the ChunkLoader is tracking.

    Parameters
    ----------
    layer : Layer
        The layer we are loading chunks for.
    """

    def __init__(self, layer):
        self.layer_id: int = id(layer)
        self.layer_ref: weakref.ReferenceType = weakref.ref(layer)

    def get_layer(self):
        """Resolve our weakref to get the layer, log if not found.

        Returns
        -------
        layer : Layer
            The layer for this ChunkRequest.
        """
        layer = self.layer_ref()
        if layer is None:
            LOGGER.debug(
                "LayerInfo.get_layer: layer %d was deleted", self.layer_id
            )
        return layer
