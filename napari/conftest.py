import warnings
from typing import List

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from napari import Viewer
from napari.layers import Image, Points
from napari.plugins._builtins import napari_write_image, napari_write_points
from napari.utils import io


def pytest_addoption(parser):
    """An option to show viewers during tests. (Hidden by default).

    Showing viewers decreases test speed by about %18.  Note, due to the
    placement of this conftest.py file, you must specify the napari folder (in
    the pytest command) to use this flag.

    Example
    -------
    $ pytest napari --show-viewer
    """
    parser.addoption(
        "--show-viewer",
        action="store_true",
        default=False,
        help="don't show viewer during tests",
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
def viewer_factory(qtbot, request):
    viewers: List[Viewer] = []

    def actual_factory(*model_args, **model_kwargs):
        model_kwargs['show'] = model_kwargs.pop(
            'show', request.config.getoption("--show-viewer")
        )
        viewer = Viewer(*model_args, **model_kwargs)
        viewers.append(viewer)
        view = viewer.window.qt_viewer
        return view, viewer

    yield actual_factory

    for viewer in viewers:
        viewer.close()


@pytest.fixture(params=['image', 'points', 'points-with-properties'])
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
            return (
                io.imread(path),
                {},  # metadata
            )

    elif request.param == 'points':
        data = np.random.rand(20, 2)
        Layer = Points
        layer = Points(data)
        writer = napari_write_points
        extension = '.csv'

        def reader(path):
            return (
                io.read_csv(path)[0][:, 1:3],
                {},  # metadata
            )

    elif request.param == 'points-with-properties':
        data = np.random.rand(20, 2)
        Layer = Points
        layer = Points(data, properties={'values': np.random.rand(20)})
        writer = napari_write_points
        extension = '.csv'

        def reader(path):
            return (
                io.read_csv(path)[0][:, 1:3],
                {
                    'properties': {
                        io.read_csv(path)[1][3]: io.read_csv(path)[0][:, 3]
                    }
                },
            )

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
    layer_data = [l.as_layer_data_tuple() for l in layers]
    layer_types = [layer._type_string for layer in layers]
    filenames = [l.name + e for l, e in zip(layers, extensions)]
    return layers, layer_data, layer_types, filenames
