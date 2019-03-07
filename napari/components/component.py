from collections import defaultdict

from contextlib import contextmanager


class Component:


    def __init__(self):
        super().__init__()

        self._listeners = defaultdict(list)
        self._ignore_notifications = False


    def add_listener(self, type, listener):
        self._listeners[type].append(listener)


    def remove_listener(self, type, listener):
        self._listeners[type].remove(listener)


    def _notify_listeners(self, type, **kwargs):
        if not self._ignore_notifications:
            for listener in self._listeners[type]:
                listener(**kwargs)

    @contextmanager
    def ignore_notifications(self):
        self._ignore_notifications = True
        yield
        self._ignore_notifications = False