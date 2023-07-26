import numpy as np
from pydantic import Field

from napari.utils.events import EventedModel


def np_generator(ndim: int = 3):
    data = np.zeros((ndim, ndim))
    for i in range(ndim):
        data[i, ndim - i - 1] = 1
    return data


def empty(event):
    pass


class Model(EventedModel):
    a: int = 3
    b: float = 2.0
    c: np.ndarray = Field(default_factory=np_generator)

    @property
    def c_inv(self):
        return np.linalg.inv(self.c)

    @c_inv.setter
    def c_inv(self, value):
        self.c = np.linalg.inv(value)

    @property
    def d(self):
        return self.a**self.b

    @d.setter
    def d(self, value):
        self.a = value
        self.b = value / 2
        self.c = np_generator(value)

    @property
    def e(self):
        return (self.c + self.a) ** self.b

    @property
    def f(self):
        return self.c_inv**self.a


class EventedModelSuite:
    """Benchmarks for EventedModel."""

    def setup(self):
        self.model = Model()
        self.model.events.a.connect(empty)
        self.model.events.b.connect(empty)
        self.model.events.c.connect(empty)
        self.model.events.c_inv.connect(empty)
        self.model.events.e.connect(empty)
        self.model.events.f.connect(empty)

    def time_event_firing(self):
        self.model.d = 4
