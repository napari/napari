import numpy as np
from pydantic import Field

from napari.utils.events import EventedModel


def np_generator(ndim: int = 3):
    data = np.zeros((ndim, ndim))
    for i in range(ndim):
        data[i, ndim - i - 1] = 1
    return data


class Sample1(EventedModel):
    a: int = 3
    b: float = 2.0
    c: np.ndarray = Field(default_factory=np_generator)

    @property
    def c_inv(self):
        return np.linalg.inv(self.b)

    @c_inv.setter
    def c_inv(self, value):
        self.b = np.linalg.inv(value)

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


def empty(event):
    pass


def time_sample1():
    s = Sample1()
    s.events.a.connect(empty)
    s.events.b.connect(empty)
    s.events.c.connect(empty)
    s.events.c_inv.connect(empty)
    s.events.e.connect(empty)
    s.events.f.connect(empty)
    s.d = 4
