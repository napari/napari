try:
    __import__('dotenv').load_dotenv()
except ModuleNotFoundError:
    pass

import os
from functools import partial
from multiprocessing.pool import ThreadPool
from unittest.mock import MagicMock, patch

import dask.threaded
import numpy as np
import pooch
import pytest
from IPython.core.history import HistoryManager
from npe2 import DynamicPlugin, PluginManager

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

if not hasattr(pooch.utils, 'file_hash'):
    setattr(pooch.utils, 'file_hash', pooch.hashes.file_hash)

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

    --aysnc_only
        Run only asynchronous tests, not sync ones.

    Notes
    -----
    Due to the placement of this conftest.py file, you must specifically name
    the napari folder such as "pytest napari --aysnc_only"
    """

    parser.addoption(
        "--async_only",
        action="store_true",
        default=False,
        help="run only asynchronous tests",
    )


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
    if request.config.getoption("--async_only"):
        # Late import so we don't import experimental code unless using it.
        from napari.components.experimental.chunk import synchronous_loading

        with synchronous_loading(False):
            yield
    else:
        yield  # Sync so do nothing.


def _is_async_mode() -> bool:
    """Return True if we are currently loading chunks asynchronously

    Returns
    -------
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


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch):
    from napari import settings
    from napari.settings import NapariSettings

    # prevent the developer's config file from being used if it exists
    cp = NapariSettings.__private_attributes__['_config_path']
    monkeypatch.setattr(cp, 'default', None)

    # calling save() with no config path is normally an error
    # here we just have save() return if called without a valid path
    NapariSettings.__original_save__ = NapariSettings.save

    def _mock_save(self, path=None, **dict_kwargs):
        if not (path or self.config_path):
            return
        NapariSettings.__original_save__(self, path, **dict_kwargs)

    monkeypatch.setattr(NapariSettings, 'save', _mock_save)

    # this makes sure that we start with fresh settings for every test.
    settings._SETTINGS = None
    yield


@pytest.fixture(autouse=True)
def auto_shutdown_dask_threadworkers():
    """
    This automatically shutdown dask thread workers.

    We don't assert the number of threads in unchanged as other things
    modify the number of threads.
    """
    assert dask.threaded.default_pool is None
    try:
        yield
    finally:
        if isinstance(dask.threaded.default_pool, ThreadPool):
            dask.threaded.default_pool.close()
            dask.threaded.default_pool.join()
        elif dask.threaded.default_pool:
            dask.threaded.default_pool.shutdown()
        dask.threaded.default_pool = None


# this is not the proper way to configure IPython, but it's an easy one.
# This will prevent IPython to try to write history on its sql file and do
# everything in memory.
# 1) it saves a thread and
# 2) it can prevent issues with slow or read-only file systems in CI.
HistoryManager.enabled = False


@pytest.fixture
def napari_svg_name():
    """the plugin name changes with npe2 to `napari-svg` from `svg`."""
    from importlib.metadata import metadata

    if tuple(metadata('napari-svg')['Version'].split('.')) < ('0', '1', '6'):
        return 'svg'
    else:
        return 'napari-svg'


@pytest.fixture(autouse=True, scope='session')
def _no_error_reports():
    """Turn off napari_error_reporter if it's installed."""
    try:
        p1 = patch('napari_error_reporter.capture_exception')
        p2 = patch('napari_error_reporter.install_error_reporter')
        with p1, p2:
            yield
    except (ModuleNotFoundError, AttributeError):
        yield


@pytest.fixture
def tmp_reader():
    """Return a temporary reader registered with the given plugin manager."""

    def make_plugin(
        pm, name, filename_patterns=['*.fake'], accepts_directories=False
    ):
        reader_plugin = DynamicPlugin(name, plugin_manager=pm)

        @reader_plugin.contribute.reader(
            filename_patterns=filename_patterns,
            accepts_directories=accepts_directories,
        )
        def read_func(pth):
            ...

        reader_plugin.register()
        return reader_plugin

    return make_plugin


@pytest.fixture
def mock_npe2_pm():
    """Mock plugin manager with no registered plugins."""
    mock_reg = MagicMock()
    with patch.object(PluginManager, 'discover'):
        _pm = PluginManager(reg=mock_reg)
    with patch('npe2.PluginManager.instance', return_value=_pm):
        yield _pm
