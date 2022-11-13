import contextlib
import sys
import time
from typing import List

import numpy as np
import pytest
from magicgui import magicgui

from napari import Viewer, types
from napari._tests.utils import layer_test_data
from napari.layers import Image, Labels, Layer
from napari.utils._proxies import PublicOnlyProxy
from napari.utils.misc import all_subclasses

try:
    import qtpy  # noqa
except ModuleNotFoundError:
    pytest.skip('Cannot test magicgui without qtpy.', allow_module_level=True)
except RuntimeError:
    pytest.skip(
        'Cannot test magicgui without Qt bindings.', allow_module_level=True
    )


# only test the first of each layer type
test_data = []
for cls in all_subclasses(Layer):
    # OctTree Image doesn't have layer_test_data
    with contextlib.suppress(StopIteration):
        test_data.append(next(x for x in layer_test_data if x[0] is cls))
test_data.sort(key=lambda x: x[0].__name__)  # required for xdist to work


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
    def add_data() -> dtype:  # type: ignore
        # and data is just the bare numpy-array or similar
        return data

    viewer.window.add_dock_widget(add_data)
    add_data()
    assert len(viewer.layers) == 1
    assert isinstance(viewer.layers[0], LayerType)
    assert viewer.layers[0].source.widget == add_data


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason='Futures not subscriptable before py3.9'
)
@pytest.mark.parametrize('LayerType, data, ndim', test_data)
def test_magicgui_add_future_data(
    qtbot, make_napari_viewer, LayerType, data, ndim
):
    """Test that annotating with Future[] works."""
    from concurrent.futures import Future
    from functools import partial

    from qtpy.QtCore import QTimer

    viewer = make_napari_viewer()
    dtype = getattr(types, f'{LayerType.__name__}Data')

    @magicgui
    # where `dtype` is something like napari.types.ImageData
    def add_data() -> Future[dtype]:  # type: ignore
        future = Future()
        # simulate something that isn't immediately ready when function returns
        QTimer.singleShot(10, partial(future.set_result, data))
        return future

    viewer.window.add_dock_widget(add_data)

    def _assert_stuff():
        assert len(viewer.layers) == 1
        assert isinstance(viewer.layers[0], LayerType)
        assert viewer.layers[0].source.widget == add_data

    assert len(viewer.layers) == 0
    with qtbot.waitSignal(viewer.layers.events.inserted):
        add_data()
    _assert_stuff()


@pytest.mark.sync_only
def test_magicgui_add_threadworker(qtbot, make_napari_viewer):
    """Test that annotating with FunctionWorker works."""
    from napari.qt.threading import FunctionWorker, thread_worker

    viewer = make_napari_viewer()
    DATA = np.random.rand(10, 10)

    @magicgui
    def add_data(x: int) -> FunctionWorker[types.ImageData]:
        @thread_worker(start_thread=False)
        def _slow():
            time.sleep(0.1)
            return DATA

        return _slow()

    viewer.window.add_dock_widget(add_data)

    assert len(viewer.layers) == 0
    worker = add_data()
    # normally you wouldn't start the worker outside of the mgui function
    # this is just to make testing with threads easier
    with qtbot.waitSignal(worker.finished):
        worker.start()

    assert len(viewer.layers) == 1
    assert isinstance(viewer.layers[0], Image)
    assert viewer.layers[0].source.widget == add_data
    assert np.array_equal(viewer.layers[0].data, DATA)


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


def test_magicgui_add_layer_list(make_napari_viewer):
    viewer = make_napari_viewer()

    @magicgui
    def add_layer() -> List[Layer]:
        a = Image(data=np.random.randint(0, 10, size=(10, 10)))
        b = Labels(data=np.random.randint(0, 10, size=(10, 10)))
        return [a, b]

    viewer.window.add_dock_widget(add_layer)
    add_layer()
    assert len(viewer.layers) == 2
    assert isinstance(viewer.layers[0], Image)
    assert isinstance(viewer.layers[1], Labels)

    assert viewer.layers[0].source.widget == add_layer
    assert viewer.layers[1].source.widget == add_layer


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
    # Make two DIFFERENT viewers
    viewer1 = make_napari_viewer()
    viewer2 = make_napari_viewer()
    assert viewer2 is not viewer1
    # Ensure one is returned by napari.current_viewer()
    from napari import current_viewer

    assert current_viewer() is viewer2

    @magicgui
    def func(v: Viewer):
        return v

    def func_returns(v: Viewer) -> bool:
        """Helper function determining whether func() returns v"""
        func_viewer = func()
        assert isinstance(func_viewer, PublicOnlyProxy)
        return func_viewer.__wrapped__ is v

    # We expect func's Viewer to be current_viewer, not viewer
    assert func_returns(viewer2)
    assert not func_returns(viewer1)
    # With viewer as parent, it should be returned instead
    viewer1.window.add_dock_widget(func)
    assert func_returns(viewer1)
    assert not func_returns(viewer2)
    # no widget should be shown
    assert not func.v.visible
    # ensure that viewer2 is still the current viewer
    assert current_viewer() is viewer2


def test_layers_populate_immediately(make_napari_viewer):
    """make sure that the layers dropdown is populated upon adding to viewer"""
    from magicgui.widgets import create_widget

    labels_layer = create_widget(annotation=Labels, label="ROI")
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((10, 10), dtype=int))
    assert not len(labels_layer.choices)
    viewer.window.add_dock_widget(labels_layer)
    assert len(labels_layer.choices) == 1
