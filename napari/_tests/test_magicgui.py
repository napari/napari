import contextlib
import sys
import time
from enum import Enum
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest
from magicgui import magicgui

from napari import Viewer, layers, types
from napari._tests.utils import layer_test_data
from napari.layers import Image, Labels, Layer, Points, Surface
from napari.utils._magicgui import _get_ndim_from_data
from napari.utils._proxies import PublicOnlyProxy
from napari.utils.migrations import _DeprecatingDict
from napari.utils.misc import all_subclasses

if TYPE_CHECKING:
    import napari.types

try:
    import qtpy  # noqa: F401 need to be ignored as qtpy may be available but Qt bindings may not be
except ModuleNotFoundError:
    pytest.skip('Cannot test magicgui without qtpy.', allow_module_level=True)
except RuntimeError:
    pytest.skip(
        'Cannot test magicgui without Qt bindings.', allow_module_level=True
    )


@pytest.fixture
def image_layer():
    return Image(
        np.empty((40, 40), dtype=np.uint8),
        scale=(1, 2),
        translate=(3, 4),
        units=('nm', 'um'),
    )


@pytest.fixture
def image_layer3d():
    return Image(
        np.empty((10, 40, 40), dtype=np.uint8),
        scale=(1, 2, 2),
        translate=(3, 4, 4),
        units=('nm', 'um', 'um'),
    )


@pytest.fixture
def image_layer_rgb():
    return Image(
        np.empty((40, 40, 3), dtype=np.uint8),
        scale=(1, 2),
        translate=(3, 4),
        units=('nm', 'um'),
    )


@pytest.fixture
def labels_layer():
    return Labels(
        np.empty((40, 40), dtype=np.uint8),
        scale=(1, 2),
        translate=(3, 4),
        units=('nm', 'um'),
    )


@pytest.fixture
def points_layer():
    return Points(
        np.empty((10, 2), dtype=np.uint8).astype(np.float32),
        scale=(1, 2),
        translate=(3, 4),
        units=('nm', 'um'),
    )


@pytest.fixture
def surface_layer():
    rng = np.random.default_rng(0)
    return Surface(
        (20 * rng.random((10, 3)), rng.integers(10, size=(6, 3))),
        scale=(1, 2, 2),
        translate=(3, 4, 4),
        units=('nm', 'um', 'um'),
    )


@pytest.fixture(
    params=['image_layer', 'labels_layer', 'points_layer', 'image_layer_rgb']
)
def layer_and_type(request):
    data = request.getfixturevalue(request.param)
    return data, getattr(types, f'{data.__class__.__name__}Data')


# only test the first of each layer type
test_data = []
for cls in all_subclasses(Layer):
    # OctTree Image doesn't have layer_test_data
    with contextlib.suppress(StopIteration):
        test_data.append(next(x for x in layer_test_data if x[0] is cls))
test_data.sort(key=lambda x: x[0].__name__)  # required for xdist to work


@pytest.mark.parametrize(('LayerType', 'data', 'ndim'), layer_test_data)
def test_get_ndim_from_data(LayerType, data, ndim):
    assert _get_ndim_from_data(data, LayerType.__name__.lower()) == ndim


@pytest.mark.parametrize(('LayerType', 'data', 'ndim'), test_data)
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


def test_magicgui_add_data_inheritance(
    make_napari_viewer, layer_and_type, image_layer
):
    """This test validates if the scale and translate are inherited from the
    previous layer when adding a new layer with magicgui if function requests,
    a LayerData type.
    """
    layer, type_ = layer_and_type
    viewer = make_napari_viewer()
    viewer.add_layer(image_layer)

    @magicgui
    def add_data(data: types.ImageData) -> type_:
        return layer.data

    viewer.window.add_dock_widget(add_data)
    add_data()
    assert len(viewer.layers) == 2
    assert isinstance(viewer.layers[1], layer.__class__)
    npt.assert_array_equal(viewer.layers[1].scale, (1, 2))
    npt.assert_array_equal(viewer.layers[1].translate, (3, 4))
    assert viewer.layers[1].units == viewer.layers[0].units


def test_magicgui_add_data_inheritance_surface(
    make_napari_viewer, surface_layer, image_layer3d
):
    """This test validates if the scale and translate are inherited from the
    previous layer when adding a new layer with magicgui if function requests,
    a LayerData type.
    """
    viewer = make_napari_viewer()
    viewer.add_layer(image_layer3d)

    @magicgui
    def add_data(data: types.ImageData) -> types.SurfaceData:
        return surface_layer.data

    viewer.window.add_dock_widget(add_data)
    add_data()
    assert len(viewer.layers) == 2
    assert isinstance(viewer.layers[1], Surface)
    npt.assert_array_equal(viewer.layers[1].scale, (1, 2, 2))
    npt.assert_array_equal(viewer.layers[1].translate, (3, 4, 4))
    assert viewer.layers[1].units == viewer.layers[0].units


def test_magicgui_add_data_inheritance_two_layer(make_napari_viewer):
    """This test validates if the scale and translate are inherited if more than
    one source layer is passed to function when adding a new layer
    with magicgui if function requests, a LayerData type.
    """
    rng = np.random.default_rng(0)
    viewer = make_napari_viewer()
    viewer.add_image(rng.random((10, 10)), scale=(1, 2), translate=(3, 4))
    viewer.add_labels(
        (rng.random((10, 10)) > 0.5).astype('uint8'),
        scale=(1, 2),
        translate=(3, 4),
    )

    @magicgui
    def add_data(
        data1: types.ImageData, data2: types.LabelsData
    ) -> types.LabelsData:
        return (data1 * data2).astype('uint8')

    viewer.window.add_dock_widget(add_data)
    add_data()
    assert len(viewer.layers) == 3
    assert isinstance(viewer.layers[2], Labels)
    npt.assert_array_equal(viewer.layers[2].scale, (1, 2))
    npt.assert_array_equal(viewer.layers[2].translate, (3, 4))


def test_magicgui_add_data_inheritance_two_layer_inconsistent(
    make_napari_viewer, monkeypatch
):
    """This test validates the scale and translate are not inherited from the
    previous layers with inconsistend metadata when adding a new layer
    with magicgui if function requests,
    a LayerData type.
    """
    rng = np.random.default_rng(0)
    viewer = make_napari_viewer()
    viewer.add_image(rng.random((10, 10)), scale=(1, 2), translate=(3, 4))
    viewer.add_labels(
        (rng.random((10, 10)) > 0.5).astype('uint8'),
        scale=(2, 2),
        translate=(3, 4),
    )

    @magicgui
    def add_data(
        data1: types.ImageData, data2: types.LabelsData
    ) -> types.LabelsData:
        return (data1 * data2).astype('uint8')

    viewer.window.add_dock_widget(add_data)
    mock = Mock()
    monkeypatch.setattr(
        'napari.utils.notifications.notification_manager.dispatch', mock
    )
    add_data()
    mock.assert_called_once()
    assert mock.call_args[0][0].message.startswith('Cannot inherit spatial')
    assert len(viewer.layers) == 3
    assert isinstance(viewer.layers[2], Labels)
    npt.assert_array_equal(viewer.layers[2].scale, (1, 1))
    npt.assert_array_equal(viewer.layers[2].translate, (0, 0))


def test_magicgui_add_layer_inheritance(make_napari_viewer):
    """This test validates if the scale and translate are inherited from the
    previous layer when adding a new layer with magicgui if function requests,
    a Layer type.
    It also checks if the presence of additional combo box in the
    function does not affect getting the data from the previous layer.
    """
    rng = np.random.default_rng(0)
    viewer = make_napari_viewer()
    viewer.add_image(rng.random((10, 10)), scale=(2, 2), translate=(1, 1))

    class SampleEnum(Enum):
        A = 'A'
        B = 'B'

    @magicgui
    def add_data(
        data: Image, factor: float = 0.5, combo: SampleEnum = SampleEnum.A
    ) -> types.LabelsData:
        return (data.data > factor).astype(
            'uint8' if combo == SampleEnum.A else 'uint16'
        )

    viewer.window.add_dock_widget(add_data)
    add_data()
    assert len(viewer.layers) == 2
    assert isinstance(viewer.layers[1], Labels)
    npt.assert_array_equal(viewer.layers[1].scale, (2, 2))
    npt.assert_array_equal(viewer.layers[1].translate, (1, 1))


def test_magicgui_add_data_inheritance_upper_dim(make_napari_viewer):
    """In the current implementation, layer with dimensionality lower than produced data are ignored"""
    rng = np.random.default_rng(0)
    viewer = make_napari_viewer()
    viewer.add_image(rng.random((10, 10)), scale=(2, 2), translate=(1, 1))

    @magicgui
    def add_data(data: types.ImageData) -> types.LabelsData:
        return np.stack([(data > 0.5), (data > 0.4)]).astype('uint8')

    viewer.window.add_dock_widget(add_data)
    add_data()
    assert len(viewer.layers) == 2
    assert isinstance(viewer.layers[1], Labels)
    npt.assert_array_equal(viewer.layers[1].scale, (1, 1, 1))
    npt.assert_array_equal(viewer.layers[1].translate, (0, 0, 0))


def test_magicgui_add_data_inheritance_less_dim(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(
        np.random.rand(4, 10, 10), scale=(1, 2, 2), translate=(2, 1, 1)
    )

    @magicgui
    def add_data(data: types.ImageData) -> types.LabelsData:
        return (data[0] > 0.5).astype('uint8')

    viewer.window.add_dock_widget(add_data)
    add_data()
    assert len(viewer.layers) == 2
    assert isinstance(viewer.layers[1], Labels)
    npt.assert_array_equal(viewer.layers[1].scale, (2, 2))
    npt.assert_array_equal(viewer.layers[1].translate, (1, 1))


def test_add_layer_data_to_viewer_optional(make_napari_viewer):
    viewer = make_napari_viewer()

    @magicgui
    def func_optional(a: bool) -> 'napari.types.ImageData | None':
        if a:
            return np.zeros((10, 10))
        return None

    viewer.window.add_dock_widget(func_optional)
    assert not viewer.layers

    func_optional(a=True)

    assert len(viewer.layers) == 1

    func_optional(a=False)

    assert len(viewer.layers) == 1


@pytest.mark.parametrize(('LayerType', 'data', 'ndim'), test_data)
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


@pytest.mark.parametrize(('LayerType', 'data', 'ndim'), test_data)
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


@pytest.mark.parametrize(('LayerType', 'data', 'ndim'), test_data)
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
    def add_layer() -> list[Layer]:
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
    def add_layer() -> list[types.LayerDataTuple]:
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


MGUI_EXPORTS = ['napari.layers.Layer', 'napari.Viewer']
MGUI_EXPORTS += [f'napari.types.{nm.title()}Data' for nm in layers.NAMES]
NAMES = ('Image', 'Labels', 'Layer', 'Points', 'Shapes', 'Surface')


@pytest.mark.parametrize('name', sorted(MGUI_EXPORTS))
def test_mgui_forward_refs(name, monkeypatch):
    """make sure that magicgui's `get_widget_class` returns the right widget type
    for the various napari types... even when expressed as strings.
    """
    import magicgui.widgets
    from magicgui.type_map import get_widget_class

    monkeypatch.delitem(sys.modules, 'napari')
    monkeypatch.delitem(sys.modules, 'napari.viewer')
    monkeypatch.delitem(sys.modules, 'napari.types')
    monkeypatch.setattr(
        'napari.utils.action_manager.action_manager._actions', {}
    )
    # need to clear all of these submodules too, otherwise the layers are oddly not
    # subclasses of napari.layers.Layer, and napari.layers.NAMES
    # oddly ends up as an empty set
    for m in list(sys.modules):
        if m.startswith('napari.layers') and 'utils' not in m:
            monkeypatch.delitem(sys.modules, m)

    wdg, options = get_widget_class(annotation=name)
    if name == 'napari.Viewer':
        assert wdg == magicgui.widgets.EmptyWidget
        assert 'bind' in options
    else:
        assert wdg == magicgui.widgets.Combobox


def test_layers_populate_immediately(make_napari_viewer):
    """make sure that the layers dropdown is populated upon adding to viewer"""
    from magicgui.widgets import create_widget

    labels_layer = create_widget(annotation=Labels, label='ROI')
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((10, 10), dtype=int))
    assert not len(labels_layer.choices)
    viewer.window.add_dock_widget(labels_layer)
    assert len(labels_layer.choices) == 1


def test_from_layer_data_tuple_accept_deprecating_dict(make_napari_viewer):
    """Test that a function returning a layer data tuple runs without error."""
    viewer = make_napari_viewer()

    @magicgui
    def from_layer_data_tuple() -> types.LayerDataTuple:
        data = np.zeros((10, 10))
        meta = _DeprecatingDict({'name': 'test_image'})
        layer_type = 'image'
        return data, meta, layer_type

    viewer.window.add_dock_widget(from_layer_data_tuple)
    from_layer_data_tuple()
    assert len(viewer.layers) == 1
    assert isinstance(viewer.layers[0], Image)
    assert viewer.layers[0].name == 'test_image'
