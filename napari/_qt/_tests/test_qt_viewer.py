import os
from unittest import mock

import numpy as np
import pytest

from napari.utils.io import imread
from napari._tests.utils import (
    add_layer_by_type,
    check_viewer_functioning,
    layer_test_data,
)


def test_qt_viewer(viewer_factory):
    """Test instantiating viewer."""
    view, viewer = viewer_factory()

    assert viewer.title == 'napari'
    assert view.viewer == viewer
    # Check no console is present before it is requested
    assert view._console is None

    assert len(viewer.layers) == 0
    assert view.layers.vbox_layout.count() == 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_qt_viewer_with_console(viewer_factory):
    """Test instantiating console from viewer."""
    view, viewer = viewer_factory()
    # Check no console is present before it is requested
    assert view._console is None
    # Check console is created when requested
    assert view.console is not None
    assert view.dockConsole.widget == view.console


def test_qt_viewer_toggle_console(viewer_factory):
    """Test instantiating console from viewer."""
    view, viewer = viewer_factory()
    # Check no console is present before it is requested
    assert view._console is None
    # Check console has been created when it is supposed to be shown
    view.toggle_console_visibility(None)
    assert view._console is not None
    assert view.dockConsole.widget == view.console


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_add_layer(viewer_factory, layer_class, data, ndim):
    view, viewer = viewer_factory(ndisplay=ndim)

    add_layer_by_type(viewer, layer_class, data)
    check_viewer_functioning(viewer, view, data, ndim)


def test_new_labels(viewer_factory):
    """Test adding new labels layer."""
    # Add labels to empty viewer
    view, viewer = viewer_factory()

    viewer._new_labels()
    assert np.max(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Add labels with image already present
    view, viewer = viewer_factory()

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer._new_labels()
    assert np.max(viewer.layers[1].data) == 0
    assert len(viewer.layers) == 2
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_new_points(viewer_factory):
    """Test adding new points layer."""
    # Add labels to empty viewer
    view, viewer = viewer_factory()

    viewer.add_points()
    assert len(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Add points with image already present
    view, viewer = viewer_factory()

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer.add_points()
    assert len(viewer.layers[1].data) == 0
    assert len(viewer.layers) == 2
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_new_shapes_empty_viewer(viewer_factory):
    """Test adding new shapes layer."""
    # Add labels to empty viewer
    view, viewer = viewer_factory()

    viewer.add_shapes()
    assert len(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Add points with image already present
    view, viewer = viewer_factory()

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer.add_shapes()
    assert len(viewer.layers[1].data) == 0
    assert len(viewer.layers) == 2
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_screenshot(viewer_factory):
    "Test taking a screenshot"
    view, viewer = viewer_factory()

    np.random.seed(0)
    # Add image
    data = np.random.random((10, 15))
    viewer.add_image(data)

    # Add labels
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)

    # Add points
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)

    # Add vectors
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)

    # Add shapes
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)

    # Take screenshot
    screenshot = view.screenshot()
    assert screenshot.ndim == 3


def test_screenshot_dialog(viewer_factory, tmpdir):
    """Test save screenshot functionality."""
    view, viewer = viewer_factory()

    np.random.seed(0)
    # Add image
    data = np.random.random((10, 15))
    viewer.add_image(data)

    # Add labels
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)

    # Add points
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)

    # Add vectors
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)

    # Add shapes
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)

    # Save screenshot
    input_filepath = os.path.join(tmpdir, 'test-save-screenshot')
    mock_return = (input_filepath, '')
    with mock.patch('napari._qt.qt_viewer.QFileDialog') as mocker:
        mocker.getSaveFileName.return_value = mock_return
        view._screenshot_dialog()
    # Assert behaviour is correct
    expected_filepath = input_filepath + '.png'  # add default file extension
    assert os.path.exists(expected_filepath)
    output_data = imread(expected_filepath)
    expected_data = view.screenshot()
    assert np.allclose(output_data, expected_data)


@pytest.mark.parametrize(
    "dtype", ['int8', 'uint8', 'int16', 'uint16', 'float32']
)
def test_qt_viewer_data_integrity(viewer_factory, dtype):
    """Test that the viewer doesn't change the underlying array."""

    image = np.random.rand(10, 32, 32)
    image *= 200 if dtype.endswith('8') else 2 ** 14
    image = image.astype(dtype)
    imean = image.mean()

    view, viewer = viewer_factory()

    viewer.add_image(image.copy())
    datamean = viewer.layers[0].data.mean()
    assert datamean == imean
    # toggle dimensions
    viewer.dims.ndisplay = 3
    datamean = viewer.layers[0].data.mean()
    assert datamean == imean
    # back to 2D
    viewer.dims.ndisplay = 2
    datamean = viewer.layers[0].data.mean()
    assert datamean == imean
