import os
import warnings
from functools import partial
from typing import List

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from napari import Viewer
from napari.components import LayerList
from napari.layers import Image, Labels, Points, Shapes, Vectors
from napari.plugins._builtins import (
    napari_write_image,
    napari_write_labels,
    napari_write_points,
    napari_write_shapes,
)
from napari.utils import io
from napari.utils.config import async_loading

try:
    from skimage.data import image_fetcher
except ImportError:
    from skimage.data import data_dir

    class image_fetcher:
        def fetch(data_name):
            if data_name.startswith("data/"):
                data_name = data_name[5:]
            path = os.path.join(data_dir, data_name)
            if not os.path.exists(path):
                raise ValueError(
                    f"Legacy skimage image_fetcher cannot find file: {path}"
                )
            return path


def pytest_addoption(parser):
    """Add napari specific command line options.

    --show-viewer
        Show viewers during tests, they are hidden by default. Showing viewers
        decreases test speed by around 20%.

    --perfmon-only
        Run only perfmon test.

    --aysnc_only
        Run only asynchronous tests, not sync ones.

    Notes
    -----
    Due to the placement of this conftest.py file, you must specifically name
    the napari folder such as "pytest napari --show-viewer"

    For --perfmon-only must also enable perfmon with env var:
    NAPARI_PERFMON=1 pytest napari --perfmon-only
    """
    parser.addoption(
        "--show-viewer",
        action="store_true",
        default=False,
        help="don't show viewer during tests",
    )

    parser.addoption(
        "--perfmon-only",
        action="store_true",
        default=False,
        help="run only perfmon tests",
    )

    parser.addoption(
        "--async_only",
        action="store_true",
        default=False,
        help="run only asynchronous tests",
    )


@pytest.fixture
def qtbot(qtbot):
    """A modified qtbot fixture that makes sure no widgets have been leaked."""
    initial = QApplication.topLevelWidgets()
    yield qtbot
    QApplication.processEvents()
    leaks = set(QApplication.topLevelWidgets()).difference(initial)
    # still not sure how to clean up some of the remaining vispy
    # vispy.app.backends._qt.CanvasBackendDesktop widgets...
    if any([n.__class__.__name__ != 'CanvasBackendDesktop' for n in leaks]):
        raise AssertionError(f'Widgets leaked!: {leaks}')
    if leaks:
        warnings.warn(f'Widgets leaked!: {leaks}')


@pytest.fixture(scope="function")
def make_test_viewer(qtbot, request):
    viewers: List[Viewer] = []

    def actual_factory(*model_args, viewer_class=Viewer, **model_kwargs):
        model_kwargs['show'] = model_kwargs.pop(
            'show', request.config.getoption("--show-viewer")
        )
        viewer = viewer_class(*model_args, **model_kwargs)
        viewers.append(viewer)
        return viewer

    yield actual_factory

    for viewer in viewers:
        viewer.close()


@pytest.fixture(
    params=['image', 'labels', 'points', 'points-with-properties', 'shapes']
)
def layer_writer_and_data(request):
    """Fixture that supplies layer io utilities for tests.

    Parameters
    ----------
    request : _pytest.fixtures.SubRequest
        The pytest request object

    Returns
    -------
    tuple
        ``(writer, layer_data, extension, reader, Layer)``

        - writer: a function that can write layerdata to a path
        - layer_data: the layerdata tuple for this layer
        - extension: an appropriate extension for this layer type
        - reader: a function that can read this layer type from a path and
                  returns a ``(data, meta)`` tuple.
        - Layer: the Layer class
    """
    if request.param == 'image':
        data = np.random.rand(20, 20)
        Layer = Image
        layer = Image(data)
        writer = napari_write_image
        extension = '.tif'

        def reader(path):
            return (io.imread(path), {}, 'image')  # metadata

    elif request.param == 'labels':
        data = np.random.randint(0, 16000, (32, 32), 'uint64')
        Layer = Labels
        layer = Labels(data)
        writer = napari_write_labels
        extension = '.tif'

        def reader(path):
            return (io.imread(path), {}, 'labels')  # metadata

    elif request.param == 'points':
        data = np.random.rand(20, 2)
        Layer = Points
        layer = Points(data)
        writer = napari_write_points
        extension = '.csv'
        reader = partial(io.csv_to_layer_data, require_type='points')
    elif request.param == 'points-with-properties':
        data = np.random.rand(20, 2)
        Layer = Points
        layer = Points(data, properties={'values': np.random.rand(20)})
        writer = napari_write_points
        extension = '.csv'
        reader = partial(io.csv_to_layer_data, require_type='points')
    elif request.param == 'shapes':
        np.random.seed(0)
        data = [
            np.random.rand(2, 2),
            np.random.rand(2, 2),
            np.random.rand(6, 2),
            np.random.rand(6, 2),
            np.random.rand(2, 2),
        ]
        shape_type = ['ellipse', 'line', 'path', 'polygon', 'rectangle']
        Layer = Shapes
        layer = Shapes(data, shape_type=shape_type)
        writer = napari_write_shapes
        extension = '.csv'
        reader = partial(io.csv_to_layer_data, require_type='shapes')
    else:
        return None, None, None, None, None

    layer_data = layer.as_layer_data_tuple()
    return writer, layer_data, extension, reader, Layer


@pytest.fixture
def layer_data_and_types():
    """Fixture that provides some layers and filenames

    Returns
    -------
    tuple
        ``layers, layer_data, layer_types, filenames``

        - layers: some image and points layers
        - layer_data: same as above but in LayerData form
        - layer_types: list of strings with type of layer
        - filenames: the expected filenames with extensions for the layers.
    """
    layers = [
        Image(np.random.rand(20, 20), name='ex_img'),
        Image(np.random.rand(20, 20)),
        Points(np.random.rand(20, 2), name='ex_pts'),
        Points(
            np.random.rand(20, 2), properties={'values': np.random.rand(20)}
        ),
    ]
    extensions = ['.tif', '.tif', '.csv', '.csv']
    layer_data = [layer.as_layer_data_tuple() for layer in layers]
    layer_types = [layer._type_string for layer in layers]
    filenames = [layer.name + e for layer, e in zip(layers, extensions)]
    return layers, layer_data, layer_types, filenames


@pytest.fixture(
    params=[
        'image',
        'labels',
        'points',
        'shapes',
        'shapes-rectangles',
        'vectors',
    ]
)
def layer(request):
    """Parameterized fixture that supplies a layer for testing.

    Parameters
    ----------
    request : _pytest.fixtures.SubRequest
        The pytest request object

    Returns
    -------
    napari.layers.Layer
        The desired napari Layer.
    """
    np.random.seed(0)
    if request.param == 'image':
        data = np.random.rand(20, 20)
        return Image(data)
    elif request.param == 'labels':
        data = np.random.randint(10, size=(20, 20))
        return Labels(data)
    elif request.param == 'points':
        data = np.random.rand(20, 2)
        return Points(data)
    elif request.param == 'shapes':
        data = [
            np.random.rand(2, 2),
            np.random.rand(2, 2),
            np.random.rand(6, 2),
            np.random.rand(6, 2),
            np.random.rand(2, 2),
        ]
        shape_type = ['ellipse', 'line', 'path', 'polygon', 'rectangle']
        return Shapes(data, shape_type=shape_type)
    elif request.param == 'shapes-rectangles':
        data = np.random.rand(7, 4, 2)
        return Shapes(data)
    elif request.param == 'vectors':
        data = np.random.rand(20, 2, 2)
        return Vectors(data)
    else:
        return None


@pytest.fixture()
def layers():
    """Fixture that supplies a layers list for testing.

    Returns
    -------
    napari.components.LayerList
        The desired napari LayerList.
    """
    np.random.seed(0)
    list_of_layers = [
        Image(np.random.rand(20, 20)),
        Labels(np.random.randint(10, size=(20, 2))),
        Points(np.random.rand(20, 2)),
        Shapes(np.random.rand(10, 2, 2)),
        Vectors(np.random.rand(10, 2, 2)),
    ]
    return LayerList(list_of_layers)


@pytest.fixture
def two_pngs():
    return [image_fetcher.fetch(f'data/{n}.png') for n in ('moon', 'camera')]


@pytest.fixture
def rgb_png():
    return [image_fetcher.fetch('data/astronaut.png')]


@pytest.fixture
def single_png():
    return [image_fetcher.fetch('data/camera.png')]


@pytest.fixture
def irregular_images():
    return [image_fetcher.fetch(f'data/{n}.png') for n in ('camera', 'coins')]


@pytest.fixture
def single_tiff():
    return [image_fetcher.fetch('data/multipage.tif')]


# Currently we cannot run async and async in the invocation of pytest
# because we get a segfault for unknown reasons. So for now:
# "pytest" runs sync_only
# "pytest napari --async_only" runs async only
@pytest.fixture(scope="session", autouse=True)
def configure_loading(request):
    """Configure async/async loading."""
    async_mode = request.config.getoption("--async_only")
    if async_mode:
        # Late import so we don't import experimental code unless using it.
        from napari.components.experimental.chunk import synchronous_loading

        with synchronous_loading(False):
            yield
    else:
        yield  # Sync so do nothing.


def _is_async_mode() -> bool:
    """Return True if we are currently loading chunks asynchronously

    Return
    ------
    bool
        True if we are currently loading chunks asynchronously.
    """
    if not async_loading:
        return False  # Not enabled at all.
    else:
        # Late import so we don't import experimental code unless using it.
        from napari.components.experimental.chunk import chunk_loader

        return not chunk_loader.force_synchronous


@pytest.fixture(autouse=True)
def skip_sync_only(request):
    """Skip async_only tests if running async."""
    sync_only = request.node.get_closest_marker('sync_only')
    if _is_async_mode() and sync_only:
        pytest.skip("running with --async_only")


@pytest.fixture(autouse=True)
def skip_async_only(request):
    """Skip async_only tests if running sync."""
    async_only = request.node.get_closest_marker('async_only')
    if not _is_async_mode() and async_only:
        pytest.skip("not running with --async_only")


@pytest.fixture(autouse=True)
def perfmon_only(request):
    """If flag is set, only run the perfmon tests."""
    perfmon_flag = request.config.getoption("--perfmon-only")
    perfmon_test = request.node.get_closest_marker('perfmon')
    if perfmon_flag and not perfmon_test:
        pytest.skip("running with --perfmon-only")
    if not perfmon_flag and perfmon_test:
        pytest.skip("not running with --perfmon-only")


# _PYTEST_RAISE=1 will prevent pytest from handling exceptions.
# Use with a debugger that's set to break on "unhandled exceptions".
# https://github.com/pytest-dev/pytest/issues/7409
if os.getenv('_PYTEST_RAISE', "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value
