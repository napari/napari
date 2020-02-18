import os
from unittest import mock

import numpy as np
import pytest
from skimage.io import imread

from napari._tests.utils import (
    add_layer_by_type,
    check_viewer_functioning,
    layer_test_data,
)


def test_qt_viewer(viewermodel_factory):
    """Test instantiating viewer."""
    view, viewer = viewermodel_factory()

    assert viewer.title == 'napari'
    assert view.viewer == viewer

    assert len(viewer.layers) == 0
    assert view.layers.vbox_layout.count() == 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_add_layer(viewermodel_factory, layer_class, data, ndim):
    view, viewer = viewermodel_factory(ndisplay=ndim)

    add_layer_by_type(viewer, layer_class, data)
    check_viewer_functioning(viewer, view, data, ndim)


def test_new_labels(viewermodel_factory):
    """Test adding new labels layer."""
    # Add labels to empty viewer
    view, viewer = viewermodel_factory()

    viewer._new_labels()
    assert np.max(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Add labels with image already present
    view, viewer = viewermodel_factory()

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


def test_new_points(viewermodel_factory):
    """Test adding new points layer."""
    # Add labels to empty viewer
    view, viewer = viewermodel_factory()

    viewer.add_points()
    assert len(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Add points with image already present
    view, viewer = viewermodel_factory()

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


def test_new_shapes_empty_viewer(viewermodel_factory):
    """Test adding new shapes layer."""
    # Add labels to empty viewer
    view, viewer = viewermodel_factory()

    viewer.add_shapes()
    assert len(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Add points with image already present
    view, viewer = viewermodel_factory()

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
    view.shutdown()


def test_screenshot(viewermodel_factory):
    "Test taking a screenshot"
    view, viewer = viewermodel_factory()

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
    view.shutdown()


def test_save_screenshot(viewermodel_factory, qtbot, tmpdir):
    """Test save screenshot functionality."""
    view, viewer = viewermodel_factory()
    qtbot.addWidget(view)

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
        view._save_screenshot()
    # Assert behaviour is correct
    expected_filepath = input_filepath + '.png'  # add default file extension
    assert os.path.exists(expected_filepath)
    output_data = imread(expected_filepath)
    expected_data = view.screenshot()
    assert np.allclose(output_data, expected_data)
    view.shutdown()


@pytest.mark.parametrize(
    "dtype", ['int8', 'uint8', 'int16', 'uint16', 'float32']
)
def test_qt_viewer_data_integrity(viewermodel_factory, dtype):
    """Test that the viewer doesn't change the underlying array."""

    image = np.random.rand(10, 32, 32)
    image *= 200 if dtype.endswith('8') else 2 ** 14
    image = image.astype(dtype)
    imean = image.mean()

    view, viewer = viewermodel_factory()

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
