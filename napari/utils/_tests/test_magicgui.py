import os
import sys
from typing import List

import numpy as np
import pytest
from magicgui import magicgui

from napari import Viewer, layers, types
from napari._tests.utils import layer_test_data
from napari.layers import Image, Labels, Layer
from napari.utils.misc import all_subclasses

try:
    import qtpy  # noqa
except ImportError:
    pytest.skip('Cannot test magicgui without qtpy.', allow_module_level=True)
except RuntimeError:
    pytest.skip(
        'Cannot test magicgui without Qt bindings.', allow_module_level=True
    )


if (
    os.getenv("CI")
    and sys.platform.startswith("linux")
    and sys.version_info[:2] == (3, 7)
    and qtpy.API_NAME == 'PySide2'
):
    pytest.skip(
        "magicgui tests and example tests causing segfault",
        allow_module_level=True,
    )


# only test the first of each layer type
test_data = []
for cls in all_subclasses(Layer):
    try:
        test_data.append(next(x for x in layer_test_data if x[0] is cls))
    except StopIteration:
        # OctTree Image doesn't have layer_test_data
        pass


@pytest.mark.parametrize('LayerType, data, ndim', test_data)
def test_magicgui_add_data(make_napari_viewer, LayerType, data, ndim):
    """Test that annotating with napari.types.<layer_type>Data works.

    It expects a raw data format (like a numpy array) and will add a layer
    of the corresponding type to the viewer.
    """
    viewer = make_napari_viewer()
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
    assert viewer.layers[0].source.widget == add_data


@pytest.mark.parametrize('LayerType, data, ndim', test_data)
def test_magicgui_get_data(make_napari_viewer, LayerType, data, ndim):
    """Test that annotating parameters with napari.types.<layer_type>Data.

    This will provide the same dropdown menu appearance as when annotating
    a parameter with napari.layers.<layer_type>... but the function will
    receive `layer.data` rather than `layer`
    """
    viewer = make_napari_viewer()
    dtype = getattr(types, f'{LayerType.__name__}Data')

    @magicgui
    # where `dtype` is something like napari.types.ImageData
    def add_data(x: dtype):
        # and data is just the bare numpy-array or similar
        return data

    viewer.window.add_dock_widget(add_data)
    layer = LayerType(data)
    viewer.add_layer(layer)


@pytest.mark.parametrize('LayerType, data, ndim', test_data)
def test_magicgui_add_layer(make_napari_viewer, LayerType, data, ndim):
    viewer = make_napari_viewer()

    @magicgui
    def add_layer() -> LayerType:
        return LayerType(data)

    viewer.window.add_dock_widget(add_layer)
    add_layer()
    assert len(viewer.layers) == 1
    assert isinstance(viewer.layers[0], LayerType)
    assert viewer.layers[0].source.widget == add_layer


def test_magicgui_add_layer_data_tuple(make_napari_viewer):
    viewer = make_napari_viewer()

    @magicgui
    def add_layer() -> types.LayerDataTuple:
        data = (
            np.random.randint(0, 10, size=(10, 10)),
            {'name': 'hi'},
            'labels',
        )
        # it works fine to just return `data`
        # but this will avoid mypy/linter errors and has no runtime burden
        return types.LayerDataTuple(data)

    viewer.window.add_dock_widget(add_layer)
    add_layer()
    assert len(viewer.layers) == 1
    assert isinstance(viewer.layers[0], Labels)
    assert viewer.layers[0].source.widget == add_layer


def test_magicgui_add_layer_data_tuple_list(make_napari_viewer):
    viewer = make_napari_viewer()

    @magicgui
    def add_layer() -> List[types.LayerDataTuple]:
        data1 = (np.random.rand(10, 10), {'name': 'hi'})
        data2 = (
            np.random.randint(0, 10, size=(10, 10)),
            {'name': 'hi2'},
            'labels',
        )
        return [data1, data2]  # type: ignore

    viewer.window.add_dock_widget(add_layer)
    add_layer()
    assert len(viewer.layers) == 2
    assert isinstance(viewer.layers[0], Image)
    assert isinstance(viewer.layers[1], Labels)

    assert viewer.layers[0].source.widget == add_layer
    assert viewer.layers[1].source.widget == add_layer


def test_magicgui_data_updated(make_napari_viewer):
    """Test that magic data parameters stay up to date."""
    viewer = make_napari_viewer()

    _returns = []  # the value of x returned from func

    @magicgui(auto_call=True)
    def func(x: types.PointsData):
        _returns.append(x)

    viewer.window.add_dock_widget(func)
    points = viewer.add_points(None)
    # func will have been called with an empty points
    np.testing.assert_allclose(_returns[-1], np.empty((0, 2)))
    points.add((10, 10))
    # func will have been called with 1 data including 1 point
    np.testing.assert_allclose(_returns[-1], np.array([[10, 10]]))
    points.add((15, 15))
    # func will have been called with 1 data including 2 points
    np.testing.assert_allclose(_returns[-1], np.array([[10, 10], [15, 15]]))


def test_magicgui_get_viewer(make_napari_viewer):
    """Test that annotating with napari.Viewer gets the Viewer"""
    viewer = make_napari_viewer()

    @magicgui
    def func(v: Viewer):
        return v

    assert func() is None
    viewer.window.add_dock_widget(func)
    assert func() is viewer
    # no widget should be shown
    assert not func.v.visible


MGUI_EXPORTS = ['napari.layers.Layer', 'napari.Viewer']
MGUI_EXPORTS += [f'napari.types.{nm.title()}Data' for nm in layers.NAMES]


@pytest.mark.parametrize('name', MGUI_EXPORTS)
def test_mgui_forward_refs(tmp_path, name):
    """Test magicgui forward ref annotations

    In a 'fresh' process, make sure that calling
    `magicgui.type_map.pick_widget_type` with the string version of a napari
    object triggers the appropriate imports to resolve the class in time.
    """
    import subprocess
    import textwrap

    script = """
    import magicgui
    assert magicgui.type_map._TYPE_DEFS == {{}}
    name = {0!r}
    wdg, options = magicgui.type_map.pick_widget_type(annotation=name)
    if name == 'napari.Viewer':
        assert wdg == magicgui.widgets.EmptyWidget and 'bind' in options
    else:
        assert wdg == magicgui.widgets.Combobox
    """

    script_path = tmp_path / 'script.py'
    script_path.write_text(textwrap.dedent(script.format(name)))
    subprocess.run([sys.executable, str(script_path)], check=True)
