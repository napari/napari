from qtpy.QtCore import QObject, Signal

from ..utils.chunk_loader import ChunkRequest, CHUNK_LOADER


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

    chunk_loaded_gui = Signal(ChunkRequest)

    def __init__(self):
        super().__init__()

        # ChunkLoader signals us when a chunk has been loaded.
        CHUNK_LOADER.events.chunk_loaded.connect(self._chunk_loaded_worker)

        # We signal ourself to switch things to the GUI thread (if necessary).
        self.chunk_loaded_gui.connect(self._chunk_loaded_gui)

    def _chunk_loaded_worker(self, event) -> None:
        """A chunk was loaded (worker thread)."""
        self.chunk_loaded_gui.emit(event.request)

    def _chunk_loaded_gui(self, request: ChunkRequest) -> None:
        """A chunk was loaded (gui thread).

        If this chunk's layer still exists, pass it the chunk.
        """
        layer = request.layer_ref()
        if layer is not None:
            layer.chunk_loaded(request)
