import logging


from ...utils.event_handler import EventHandler
from napari.utils.base_interface import BaseInterface


logger = logging.getLogger(__name__)


class LayerEventHandler(EventHandler):
    """
    Event handler for layer specific events. Receives change events made from
    the data/controls/ or visual interface and updates all associated
    components.

    Ex. ImageLayer components = [qt_image.py, vispy_image.py, image.py]
    """

    def __init__(self, component: BaseInterface = None):
        super().__init__(component)

    def on_change(self, event=None):
        """
        Process changes made from any interface
        """
        name = event.type
        logger.debug(f"event: {name}")
        try:  # until refactor on all layers is complete, not all events will have a value property
            value = event.value
            logger.debug(f"   value: {value}")
        except AttributeError:
            logger.debug(f"did not handle event {name}")
            return

        for component in self.components_to_update:
            update_method_name = f"_on_{name}_change"
            if hasattr(component, update_method_name):
                logger.debug(
                    f"       : {component.__class__} {update_method_name}"
                )
                update_method = getattr(component, update_method_name)
                update_method(value)
