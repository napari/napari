from abc import abstractmethod


class EventHandler:
    def __init__(self, component=None):
        self.components_to_update = [component] if component else []

    def register_component_to_update(self, component):
        self.components_to_update.append(component)

    @abstractmethod
    def on_change(self, event=None):
        ...
