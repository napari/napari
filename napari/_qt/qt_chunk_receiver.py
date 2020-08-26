"""QtChunkReceiver and QtGuiEvent classes.
"""
import logging

from qtpy.QtCore import QObject, Signal

from ..components.chunk import chunk_loader
from ..utils.events import EmitterGroup, Event

LOGGER = logging.getLogger('napari.async')


class QtGuiEvent(QObject):
    """Fires an event in the GUI thread.

    Listens to an event in any thread. When that event fires, it uses a Qt
    Signal/Slot to fire a matching even in the GUI thread. If the original
    event is already in the GUI thread that's fine, the matching event will
    be immediately fired in the GUI threads.
    """

    signal = Signal(Event)

    def __init__(self, parent, listen_event):
        super().__init__(parent)

        self.events = EmitterGroup(
            source=self, auto_connect=True, gui_event=None
        )
        listen_event.connect(self._on_event)
        self.signal.connect(self._slot)

    def _on_event(self, event) -> None:
        """Event was fired, we could be in any thread."""
        self.signal.emit(event)

    def _slot(self, event) -> None:
        """Slot is always called in the GUI thread."""
        self.events.gui_event(event=event)


class QtChunkReceiver:
    """Listens for loaded chunks, passes them to their Layer.

    Parameters
    ----------
    parent : QObject
        Parent for QtGuiEvent.
    """

    def __init__(self, parent):
        listen_event = chunk_loader.events.chunk_loaded
        self.gui_event = QtGuiEvent(parent, listen_event)
        self.gui_event.events.gui_event.connect(self._on_chunk_loaded_gui)

    def _on_chunk_loaded_gui(self, event) -> None:
        """A chunk was loaded. This method called in the GUI thread

        Parameters
        ----------
        event : Event
            The event object passed with the EmitterGroup event.
        """
        layer = event.event.layer
        request = event.event.request

        layer.on_chunk_loaded(request)  # Pass the chunk to its layer.
