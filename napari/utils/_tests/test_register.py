from dataclasses import dataclass, field

from napari.utils._register import create_func
from napari.utils.migrations import deprecated_constructor_arg_by_attr


@dataclass
class DummyClass:
    """Dummy class to test create_func"""

    layers: list = field(default_factory=list)


class SimpleClass:
    """Simple class to test create_func"""

    def __init__(self, a):
        self.a = a


class SimpleClassDeprecated:
    """Simple class to test create_func"""

    @deprecated_constructor_arg_by_attr('b')
    def __init__(self, a=1):
        self.a = a

    @property
    def b(self):
        return self.a * 2

    @b.setter
    def b(self, value):
        self.a = value // 2


def test_create_func():
    DummyClass.add_simple_class = create_func(SimpleClass)
    dc = DummyClass()
    dc.add_simple_class(a=1)
    assert dc.layers[0].a == 1


def test_create_func_deprecated():
    DummyClass.add_simple_class_deprecated = create_func(SimpleClassDeprecated)
    dc = DummyClass()
    dc.add_simple_class_deprecated(b=4)
    assert dc.layers[0].a == 2
