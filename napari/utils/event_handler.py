import logging
import weakref


logger = logging.getLogger(__name__)


class EventHandler:
    def __init__(self, component=None):
        """Event handler for controlling the flow and updates for events.

        For example. for layer specific events it receives change events made
        from the data, controls, or visual interface and updates all associated
        components.
        """
        self.components = []
        if component is not None:
            self.register_listener(component)

    def register_listener(self, component):
        """Register a component to listen to emitted events.

        Parameters
        ----------
        component : Object
            Object that contains callbacks for specific events. These are
            methods named according to an '_on_*_change' convention.
        """
        # We need to use weak references here to ensure QWigdets that are
        # registered as listeners are not leaked. See this discussion
        # https://github.com/napari/napari/pull/1391#issuecomment-653939143
        self.components.append(weakref.ref(component))

    def on_change(self, event=None):
        """Process an event from any of our event emitters.

        Parameters
        ----------
        event : napari.utils.Event
            Event emitter by any of our event emitters. Event must have a
            'type' that indicates the 'name' of the event, and a 'value'
            that carries the data associated with the event. These are
            automatically added by our event emitters.
        """
        logger.debug(f"event: {event.type}")
        # until refactor on all layers is complete, not all events will have a
        # value property
        try:
            value = event.value
            logger.debug(f" value: {value}")
        except AttributeError:
            logger.debug(f" did not handle event {event.type}")
            return

        # Update based on event value
        for componentref in self.components:
            # We use weak references here for reasons discussed inside the
            # register_listener method above
            component = componentref()
            if component is not None:
                update_method_name = f"_on_{event.type}_change"
                update_method = getattr(component, update_method_name, None)
                if update_method:
                    update_method(value)
