from contextlib import contextmanager


class Component:


    def __init__(self):
        super().__init__()

        self._listeners = []
        self._ignore_notifications = False


    def add_listener(self, listener):
        self._listeners.append(listener)


    def remove_listener(self, listener):
        self._listeners.remove(listener)


    def _notify_listeners(self, **kwargs):
        if not self._ignore_notifications:
            for listener in self._listeners:
                listener(**kwargs)

    @contextmanager
    def ignore_notifications(self):
        self._ignore_notifications = True
        yield
        self._ignore_notifications = False