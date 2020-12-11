"""RemoteMessages class.

Sends messages to remote clients.
"""
import logging
from typing import Dict

from ....layers.image.experimental.octree_image import OctreeImage
from ...layerlist import LayerList
from ..monitor import monitor, numpy_dumps

LOGGER = logging.getLogger("napari.monitor")


class RemoteMessages:
    """Sends messages to remote clients.

    Parameters
    ----------
    layers : LayerList
        The viewer's layers, so we can call into them.
    """

    def __init__(self, layers: LayerList):
        self.layers = layers
        self._frame_number = 0

    def on_poll(self) -> None:
        """Send messages to clients.

        These message go out once per frame. So it might not make sense to
        include static information that rarely changes. Although if it's
        small, maybe it's okay.

        The message looks like:

        {
            "poll": {
                "layers": {
                    13482484: {
                        "tile_state": ...
                        "tile_config": ...
                    }
                }
            }
        }
        """
        self._frame_number += 1

        layers: Dict[int, dict] = {}

        for layer in self.layers:
            if isinstance(layer, OctreeImage):
                layers[id(layer)] = layer.remote_messages

        LOGGER.info("RemoteMessages: %d", self._frame_number)
        LOGGER.info("RemoteMessages: %s", numpy_dumps(layers))
        monitor.add_data({"poll": {"layers": layers}})
