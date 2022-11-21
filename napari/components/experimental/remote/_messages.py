"""RemoteMessages class.

Sends messages to remote clients.
"""
import logging
import time
from typing import Dict

from napari.components.experimental.monitor import monitor
from napari.components.layerlist import LayerList
from napari.layers.image.experimental.octree_image import _OctreeImageBase

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
        self._last_time = None

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
            if isinstance(layer, _OctreeImageBase):
                layers[id(layer)] = layer.remote_messages

        monitor.add_data({"poll": {"layers": layers}})
        self._send_frame_time()

    def _send_frame_time(self) -> None:
        """Send the frame time since last poll."""
        now = time.time()
        last = self._last_time
        delta = now - last if last is not None else 0
        delta_ms = delta * 1000

        monitor.send_message(
            {'frame_time': {'time': now, 'delta_ms': delta_ms}}
        )
        self._last_time = now
