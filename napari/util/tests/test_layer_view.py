import numpy as np
import pytest

from napari.layers import Image, Labels, Points

from napari.util.layer_view import _baseclass_layer_to_subclass, LayerView


def test_inheritance():
    # check registered as base class
    class Foo(LayerView):
        ...

    assoc = _baseclass_layer_to_subclass[Foo]

    assert assoc == {}

    # check subclass registration
    class ImageFoo(Foo, layer=Image):
        ...

    assert assoc[Image] is ImageFoo

    # check subclass registration
    # for layer subclassing layer type present in register
    class LabelsFoo(Foo, layer=Labels):
        ...

    assert assoc[Labels] is LabelsFoo

    # check registration for subclass of registered subclass
    class PointsFoo(ImageFoo, layer=Points):
        ...

    assert assoc == {Image: ImageFoo, Labels: LabelsFoo, Points: PointsFoo}

    # check non-registration
    class NoneFoo(Foo):
        ...

    assert NoneFoo not in assoc.values()
    assert NoneFoo not in _baseclass_layer_to_subclass

    # check new base class non-interference
    class Bar(LayerView):
        ...

    assert _baseclass_layer_to_subclass[Foo] is assoc
    assert _baseclass_layer_to_subclass[Bar] == {}


def test_instantiation():
    # constants
    dummy_data = np.zeros((2, 2))
    dummy_image = Image(dummy_data)
    dummy_labels = Labels(dummy_data)
    dummy_points = Points(dummy_data)

    # should not be able to instantiate LayerView
    with pytest.raises(TypeError):
        LayerView()

    # test base class instantiation
    class Foo(LayerView):
        ...

    assert type(Foo(dummy_image)) is Foo

    # test subclass instantiation
    class ImageFoo(Foo, layer=Image):
        ...

    assert type(ImageFoo(dummy_image)) is ImageFoo

    # test subclass instantiation via base class
    assert type(Foo(dummy_image)) is ImageFoo

    # test subclass instantiation via base class
    # with multiple related layer types
    class LabelsFoo(Foo, layer=Labels):
        ...

    assert type(Foo(dummy_labels)) is LabelsFoo
    assert type(Foo(dummy_image)) is ImageFoo

    # test base class instantiation with non-registered layer type
    assert type(Foo(dummy_points)) is Foo
