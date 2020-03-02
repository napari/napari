from typing import List

from ..base import BaseInterface


class EventHandlerBase:
    """
    Base layer controller class responsible for the interactions between the data layer,
    visual rendering, and gui controls
    """

    def __init__(self, editable_components: List[BaseInterface]):
        """
        Parameters
        ----------
        editable_components:
            List of components to update that all adhere to the UpdateContractBase
            ex.  [qt_image, vispy_image, image]
        """

        self.components_to_update = editable_components
        self.connect_events()

    def connect_events(self):
        for component in self.components_to_update:
            if hasattr(component, "events"):
                for name in component.events:
                    if name == "interpolation" or name == "contrast_limits":
                        event = getattr(component.events, name)
                        event.connect(self.on_change)

    def on_change(self, event=None):
        """
        Process changes when attribute is changed from any interface
        """
        name = event.name
        value = event.value
        for component in self.components_to_update:
            update_method_name = f"_set_{name}"
            update_method = getattr(component, update_method_name)
            update_method(value)
