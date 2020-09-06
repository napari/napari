"""LoadType, LoadStats and LayerInfo.
"""
import logging
import weakref

from ...layers.base import Layer

LOGGER = logging.getLogger("napari.async")


class LayerInfo:
    """Information about one layer the ChunkLoader is tracking.

    Parameters
    ----------
    layer : Layer
        The layer we are loading chunks for.

    Attributes
    ----------
    layer_id : int
        The id of the layer.
    layer_ref : weakref
        Weak reference to the layer.
    load_type : LoadType
        Enum for whether to do auto/sync/async loads.

    Notes
    -----
    We store a weak reference because we do not want an in-progress request
    to prevent a layer from being deleted. Meanwhile, once a request has
    finished, we can de-reference the weakref to make sure the layer was
    note deleted during the load process.
    """

    def __init__(self, layer):
        self.layer_id: int = id(layer)
        self.layer_ref: weakref.ReferenceType = weakref.ref(layer)

    def get_layer(self) -> Layer:
        """Resolve our weakref to get the layer.

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
