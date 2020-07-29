from qtpy.QtCore import QObject, Signal

from ..layers.base import Layer
from ..utils.chunk import ChunkRequest, chunk_loader
from ..utils.perf import add_instant_event


class QtChunkReceiver(QObject):
    """Shuttles newly loaded chunks from the ChunkLoader to their layers.

    Notes
    -----
    The ChunkLoader's chunk_loaded event might be signaled in a worker
    thread. The documentation only guarantees it will be signaled in thread
    in this process, in theory could be any thread.

    We do not want to call into model/layer code in a worker thread so we
    signal ourselves in the GUI thread. If we are already in the GUI thread
    this does no harm and happens instantly.
    """

    chunk_loaded_gui = Signal(Layer, ChunkRequest)

    def __init__(self):
        super().__init__()

        # ChunkLoader signals us when a chunk has been loaded.
        chunk_loader.events.chunk_loaded.connect(self._chunk_loaded_worker)

        # We signal ourselves in the GUI thread.
        self.chunk_loaded_gui.connect(self._chunk_loaded_gui)

    def _chunk_loaded_worker(self, event) -> None:
        """A chunk was loaded.

        The method is probably running in a worker thread.
        """
        add_instant_event("_chunk_loaded_worker")
        self.chunk_loaded_gui.emit(event.layer, event.request)

    def _chunk_loaded_gui(self, layer, request: ChunkRequest) -> None:
        """A chunk was loaded.

        The method is definitely running in the GUI thread.
        """
        add_instant_event("_chunk_loaded_gui")

        # Pass the chunk to its layer.
        layer.chunk_loaded(request)

    def close(self):
        """Viewer is closing.
        """
        self.chunk_loaded_gui.disconnect()
        chunk_loader.events.chunk_loaded.disconnect()
