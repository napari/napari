

class Component:


    def __init__(self):
        super().__init__()

        self.listeners = []


    def add_listener(self, listener):
        self.listeners.append(listener)


    def remove_listener(self, listener):
        self.listeners.remove(listener)


    def _notify_listeners(self, **kwargs):
        for listener in self.listeners:
            listener(**kwargs)