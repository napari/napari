import os
import sys
from typing import List

import numpy as np
import pytest
from magicgui import magicgui

from napari import types
from napari._tests.utils import layer_test_data
from napari.layers import Image, Labels, Layer, Points

try:
    import qtpy  # noqa
except ImportError:
    pytest.skip('Cannot test magicgui without qtpy.', allow_module_level=True)
except RuntimeError:
    pytest.skip(
        'Cannot test magicgui without Qt bindings.', allow_module_level=True
    )


def test_magicgui_returns_image(make_test_viewer):
    """make sure a magicgui function returning Image adds an Image.

    This is deprecated and now emits a warning
    """
    viewer = make_test_viewer()

    @magicgui
    def add_image() -> Image:
        return np.random.rand(10, 10)

    viewer.window.add_dock_widget(add_image)
    assert len(viewer.layers) == 0
    with pytest.warns(UserWarning):
        add_image()  # should add a new layer to the list
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == 'add_image result'

    with pytest.warns(UserWarning):
        add_image()  # should just update existing layer on subsequent calls
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == 'add_image result'
    assert isinstance(viewer.layers[0], Image)


def test_magicgui_returns_label(make_test_viewer):
    """make sure a magicgui function returning Labels adds a Labels.

    This is deprecated and now emits a warning
    """
    viewer = make_test_viewer()

    @magicgui
    def add_labels() -> Labels:
        return np.random.rand(10, 10)

    viewer.window.add_dock_widget(add_labels)
    assert len(viewer.layers) == 0
    with pytest.warns(UserWarning):
        add_labels()  # should add a new layer to the list
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == 'add_labels result'
    assert isinstance(viewer.layers[0], Image)


@pytest.mark.skipif(
    bool(os.environ.get('CI') and sys.platform == "darwin"),
    reason="segfault on mac CI",
)
def test_magicgui_returns_layer_tuple(make_test_viewer):
    """make sure a magicgui function returning Layer adds the right type.

    This is deprecated and now emits a warning
    """
    viewer = make_test_viewer()

    @magicgui
    def add_layer() -> Layer:
        return [(np.random.rand(10, 3), {'size': 20, 'name': 'foo'}, 'points')]

    viewer.window.add_dock_widget(add_layer)
    assert len(viewer.layers) == 0

    with pytest.warns(UserWarning):
        add_layer()  # should add a new layer to the list
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'foo'
    assert isinstance(layer, Points)
    assert layer.data.shape == (10, 3)


@pytest.mark.parametrize('LayerType, data, ndim', layer_test_data)
def test_magicgui_add_data(make_test_viewer, LayerType, data, ndim):
    """Test that annotating with napari.types.<layer_type>Data works.

    It expects a raw data format (like a numpy array) and will add a layer
    of the corresponding type to the viewer.
    """
    viewer = make_test_viewer()
    dtype = getattr(types, f'{LayerType.__name__}Data')

    @magicgui
    # where `dtype` is something like napari.types.ImageData
    def add_data() -> dtype:
        # and data is just the bare numpy-array or similar
        return data

    viewer.window.add_dock_widget(add_data)
    add_data()
    assert len(viewer.layers) == 1
    assert isinstance(viewer.layers[0], LayerType)


@pytest.mark.parametrize('LayerType, data, ndim', layer_test_data)
def test_magicgui_add_layer(make_test_viewer, LayerType, data, ndim):
    viewer = make_test_viewer()

    @magicgui
    def add_layer() -> LayerType:
        return LayerType(data)

    viewer.window.add_dock_widget(add_layer)
    add_layer()
    assert len(viewer.layers) == 1
    assert isinstance(viewer.layers[0], LayerType)


def test_magicgui_add_layer_data_tuple(make_test_viewer):
    viewer = make_test_viewer()

    @magicgui
    def add_layer() -> types.LayerDataTuple:
        data = (np.random.rand(10, 10), {'name': 'hi'}, 'labels')
        # it works fine to just return `data`
        # but this will avoid mypy/linter errors and has no runtime burden
        return types.LayerDataTuple(data)

    viewer.window.add_dock_widget(add_layer)
    add_layer()
    assert len(viewer.layers) == 1
    assert isinstance(viewer.layers[0], Labels)


def test_magicgui_add_layer_data_tuple_list(make_test_viewer):
    viewer = make_test_viewer()

    @magicgui
    def add_layer() -> List[types.LayerDataTuple]:
        data1 = (np.random.rand(10, 10), {'name': 'hi'})
        data2 = (np.random.rand(10, 10), {'name': 'hi2'}, 'labels')
        return [data1, data2]  # type: ignore

    viewer.window.add_dock_widget(add_layer)
    add_layer()
    assert len(viewer.layers) == 2
    assert isinstance(viewer.layers[0], Image)
    assert isinstance(viewer.layers[1], Labels)
