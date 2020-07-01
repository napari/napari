import logging


logger = logging.getLogger(__name__)


class EventHandler:
    def __init__(self, component=None):
        """Event handler for controlling the flow and updates for events.

        For example. for layer specific events it receives change events made
        from the data, controls, or visual interface and updates all associated
        components.
        """
        self.components_to_update = (
            [component] if component is not None else []
        )

    def register_listener(self, component):
        """Register a component to listen to emitted events.

        Parameters
        ----------
        component : Object
            Object that contains callbacks for specific events. These are
            methods named according to an '_on_*_change' convention.
        """
        self.components_to_update.append(component)

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
        for component in self.components_to_update:
            update_method_name = f"_on_{event.type}_change"
            update_method = getattr(component, update_method_name, None)
            if update_method:
                update_method(value)
