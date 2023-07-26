from math import ceil

from napari.utils.events import EventedModel


def empty(event):
    pass


def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


class Model(EventedModel):
    a: int = 3
    b: float = 2.0
    c: int = 3

    @property
    def d(self):
        return (self.c + self.a) ** self.b

    @d.setter
    def d(self, value):
        self.c = value
        self.a = value
        self.b = value * 1.1

    @property
    def e(self):
        return (fibonacci(self.c) + fibonacci(self.a)) ** fibonacci(
            ceil(self.b)
        )


class EventedModelSuite:
    """Benchmarks for EventedModel."""

    def setup(self):
        self.model = Model()
        self.model.events.a.connect(empty)
        self.model.events.b.connect(empty)
        self.model.events.c.connect(empty)
        self.model.events.e.connect(empty)

    def time_event_firing(self):
        self.model.d = 4
        self.model.d = 18

    def time_long_connection(self):
        def long_connection(event):
            for _i in range(5):
                fibonacci(event.source.c)

        self.model.events.e.connect(long_connection)

        self.model.d = 15
