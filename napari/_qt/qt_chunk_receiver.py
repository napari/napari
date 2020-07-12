from qtpy.QtCore import QObject, Signal

from ..layers.base import Layer
from ..utils.chunk_loader import ChunkRequest, CHUNK_LOADER
from ..utils.perf import perf_func, timers


class QtChunkReceiver(QObject):
    """Shuttles newly loaded chunks from the ChunkLoader to layers.

    Notes
    -----
    The ChunkLoader's chunk_loaded event might be signaled in a worker
    thread. The documentation only guarantees it will be thread in this
    process, in theory could be any thread.

    We do not want to call into model/layer code in a worker thread so we
    signal ourselves in the GUI thread. If we are already in the GUI thread
    this does no harm and happens instantly.
    """

    chunk_loaded_gui = Signal(Layer, ChunkRequest)

    def __init__(self):
        super().__init__()

        # ChunkLoader signals us when a chunk has been loaded.
        CHUNK_LOADER.events.chunk_loaded.connect(self._chunk_loaded_worker)

        # We signal ourself to switch things to the GUI thread (if necessary).
        self.chunk_loaded_gui.connect(self._chunk_loaded_gui)

    @perf_func
    def _chunk_loaded_worker(self, event) -> None:
        """A chunk was loaded (worker thread)."""
        timers.add_instant_event("_chunk_loaded_worker")
        self.chunk_loaded_gui.emit(event.layer, event.request)

    @perf_func
    def _chunk_loaded_gui(self, layer, request: ChunkRequest) -> None:
        """A chunk was loaded (gui thread) pass it to the layer.
        """
        timers.add_instant_event("_chunk_loaded_gui")
        layer.chunk_loaded(request)

    def close(self):
        """Viewer is closing.
        """
        self.chunk_loaded_gui.disconnect()
        CHUNK_LOADER.events.chunk_loaded.disconnect()
