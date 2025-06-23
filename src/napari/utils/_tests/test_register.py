from dataclasses import dataclass, field

import pytest

from napari.utils._register import create_func
from napari.utils.migrations import (
    deprecated_constructor_arg_by_attr,
    rename_argument,
)


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
    @deprecated_constructor_arg_by_attr('c')
    def __init__(self, a=1):
        self.a = a

    @property
    def b(self):  # pragma: no cover
        return self.a * 2

    @b.setter
    def b(self, value):
        self.a = value // 2

    @property
    def c(self):  # pragma: no cover
        return self.a * 2

    @c.setter
    def c(self, value):
        self.a = value // 2


class SimpleClassRenamed:
    """Simple class to test create_func"""

    @rename_argument(
        from_name='c',
        to_name='a',
        version='0.6.0',
        since_version='0.4.18',
    )
    @rename_argument(
        from_name='d',
        to_name='b',
        version='0.6.0',
        since_version='0.4.18',
    )
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b


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
    dc.add_simple_class_deprecated(c=8)
    assert dc.layers[1].a == 4


def test_create_func_renamed():
    DummyClass.add_simple_class_renamed = create_func(SimpleClassRenamed)
    dc = DummyClass()
    with pytest.warns(FutureWarning, match="Argument 'c' is deprecated"):
        dc.add_simple_class_renamed(c=4)
    assert dc.layers[0].a == 4
    with pytest.warns(FutureWarning, match="Argument 'd' is deprecated"):
        dc.add_simple_class_renamed(d=8)
    assert dc.layers[1].b == 8
