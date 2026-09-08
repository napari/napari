class Signal:
    def __init__(self):
        self.cache = {}

    def __get__(self, instance, owner):
        if instance is None:
            return self

        try:
            return self.cache[instance]
        except KeyError:
            self.cache[instance] = instance = BoundSignal()
            return instance


class BoundSignal:
    def __init__(self):
        self._subscribers = []

    def emit(self, *args):
        for sub in self._subscribers:
            sub(*args)

    def connect(self, listener):
        self._subscribers.append(listener)
