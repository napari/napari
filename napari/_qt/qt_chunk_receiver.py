"""QtChunkReceiver and QtGuiEvent classes.
"""
import logging

from qtpy.QtCore import QObject, Signal

from ..components.chunk import chunk_loader
from ..utils.events import EmitterGroup, Event, EventEmitter

LOGGER = logging.getLogger('napari.async')


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
    """

    signal = Signal(Event)

    def __init__(self, parent: QObject, emitter: EventEmitter):
        super().__init__(parent)

        emitter.connect(self._on_event)
        self.emitter = emitter

        self.events = EmitterGroup(
            source=self, auto_connect=True, gui_event=None
        )

        self.signal.connect(self._slot)

    def _on_event(self, event) -> None:
        """Event was fired, we could be in any thread."""
        self.signal.emit(event)

    def _slot(self, event) -> None:
        """Slot is always called in the GUI thread."""
        self.events.gui_event(event=event)

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
    """

    def __init__(self, parent: QObject):
        listen_event = chunk_loader.events.chunk_loaded
        self.gui_event = QtGuiEvent(parent, listen_event)
        self.gui_event.events.gui_event.connect(self._on_chunk_loaded_gui)

    def _on_chunk_loaded_gui(self, event) -> None:
        """A chunk was loaded. This method is called in the GUI thread.

        Parameters
        ----------
        event : Event
            The event object from the original event.
        """
        layer = event.event.layer
        request = event.event.request

        LOGGER.info(
            "QtChunkReceiver._on_chunk_loaded_gui: data_id=%d",
            request.key.data_id,
        )

        layer.on_chunk_loaded(request)  # Pass the chunk to its layer.

    def close(self):
        """Viewer is closing."""
        self.gui_event.close()
