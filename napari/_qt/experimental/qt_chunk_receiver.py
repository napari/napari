"""QtChunkReceiver and QtGuiEvent classes.
"""
from qtpy.QtCore import QObject, Signal

from napari.components.experimental.chunk import chunk_loader
from napari.utils.events import EmitterGroup, Event, EventEmitter


class QtGuiEvent(QObject):
    """Fires an event in the GUI thread.

    Listens to an event in any thread. When that event fires, it uses a Qt
    Signal/Slot to fire a gui_event in the GUI thread. If the original
    event is already in the GUI thread that's fine, the gui_event will
    be immediately fired the GUI thread.

    Parameters
    ----------
    parent : QObject
        Parent Qt object.
    emitter : EventEmitter
        The event we are listening to.

    Attributes
    ----------
    emitter : EventEmitter
        The event we are listening to.
    events : EmitterGroup
        The only event we report is events.gui_event.

    Notes
    -----
    Qt's signal/slot mechanism is the only way we know of to "call" from a
    worker thread to the GUI thread. When Qt signals from a worker thread
    it posts a message to the GUI thread. When the GUI thread is next
    processing messages it will receive that message and call into the Slot
    to deliver the message/event.

    If the original event was already in the GUI thread that's fine,
    the resulting event will just be triggered right away.
    """

    signal = Signal(Event)

    def __init__(self, parent: QObject, emitter: EventEmitter):
        super().__init__(parent)

        emitter.connect(self._on_event)
        self.emitter = emitter

        self.events = EmitterGroup(source=self, gui_event=None)

        self.signal.connect(self._slot)

    def _on_event(self, event) -> None:
        """Event was fired, we could be in any thread."""
        self.signal.emit(event)

    def _slot(self, event) -> None:
        """Slot is always called in the GUI thread."""
        self.events.gui_event(original_event=event)

    def close(self):
        """Viewer is closing."""
        self.gui_event.disconnect()
        self.emitter.disconnect()


class QtChunkReceiver:
    """Passes loaded chunks to their layer.

    Parameters
    ----------
    parent : QObject
        Parent Qt object.

    Attributes
    ----------
    gui_event : QtGuiEvent
        We use this to call _on_chunk_loaded_gui() in the GUI thread.

    Notes
    -----
    ChunkLoader._done "may" be called in a worker thread. The
    concurrent.futures documentation only guarantees that the future's done
    handler will be called in a thread in the correct process, it does not
    say which thread.

    We need to call Layer.on_chunk_loaded() to deliver the loaded chunk to the
    Layer. We do not want to make this call from a worker thread, because our
    model code is not thread safe. We don't want the GUI thread and the worker
    thread changing things at the same time, both triggering events, potentially
    calling into vispy or other things that also aren't thread safe.

    We could add locks, but it's simpler and better if we just call
    Layer.on_chunk_loaded() from the GUI thread. This class QtChunkReceiver
    listens to the ChunkLoader's chunk_loaded event. It then uses QtUiEvent
    to call its own _on_chunk_loaded_gui() in the GUI thread. From that
    method it can safely call Layer.on_chunk_loaded.

    If ChunkLoader's chunk_loaded event is already in the GUI thread for
    some reason, this class will still work fine, it will just run
    100% in the GUI thread.
    """

    def __init__(self, parent: QObject):
        listen_event = chunk_loader.events.chunk_loaded
        self.gui_event = QtGuiEvent(parent, listen_event)
        self.gui_event.events.gui_event.connect(self._on_chunk_loaded_gui)

    @staticmethod
    def _on_chunk_loaded_gui(event) -> None:
        """A chunk was loaded. This method is called in the GUI thread.

        Parameters
        ----------
        event : Event
            The event object from the original event.
        """
        layer = event.original_event.layer
        request = event.original_event.request

        layer.on_chunk_loaded(request)  # Pass the chunk to its layer.

    def close(self):
        """Viewer is closing."""
        self.gui_event.close()
