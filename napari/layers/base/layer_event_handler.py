from ...utils.event_handler import EventHandler
from napari.utils.base_interface import BaseInterface


class LayerEventHandler(EventHandler):
    """
    Event handler for layer specific events. Receives change events made from the data/controls/
    or visual interface and updates all associated components.

    Ex. ImageLayer components = [qt_image.py, vispy_image.py, image.py]
    """

    def __init__(self, component: BaseInterface = None):
        super().__init__(component)

    def on_change(self, event=None):
        """
        Process changes made from any interface
        """
        name = event.type
        value = event.value
        print(f"event: {name}")
        for component in self.components_to_update:
            update_method_name = f"_on_{name}_change"
            update_method = getattr(component, update_method_name)
            update_method(value)
