import numpy as np
import pytest


def test_4D_5D_images(make_test_viewer):
    """Test adding 4D followed by 5D image layers to the viewer.

    Initially only 2 sliders should be present, then a third slider should be
    created.
    """
    np.random.seed(0)
    viewer = make_test_viewer()
    view = viewer.window.qt_viewer

    # add 4D image data
    data = np.random.random((2, 6, 30, 40))
    viewer.add_image(data)
    assert np.all(viewer.layers[0].data == data)
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 4
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 2

    # now add 5D image data - check an extra slider has been created
    data = np.random.random((4, 4, 5, 30, 40))
    viewer.add_image(data)
    assert np.all(viewer.layers[1].data == data)
    assert len(viewer.layers) == 2
    assert viewer.dims.ndim == 5
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 3


def test_5D_image_3D_rendering(make_test_viewer):
    """Test 3D rendering of a 5D image."""
    np.random.seed(0)
    viewer = make_test_viewer()
    view = viewer.window.qt_viewer

    # add 4D image data
    data = np.random.random((2, 10, 12, 13, 14))
    viewer.add_image(data)
    assert np.all(viewer.layers[0].data == data)
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 5
    assert viewer.dims.ndisplay == 2
    assert viewer.layers[0]._data_view.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 3

    # switch to 3D rendering
    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3
    assert viewer.layers[0]._data_view.ndim == 3
    assert np.sum(view.dims._displayed_sliders) == 2


def test_change_image_dims(make_test_viewer):
    """Test changing the dims and shape of an image layer in place and checking
    the numbers of sliders and their ranges changes appropriately.
    """
    np.random.seed(0)
    viewer = make_test_viewer()
    view = viewer.window.qt_viewer

    # add 3D image data
    data = np.random.random((10, 30, 40))
    viewer.add_image(data)
    assert np.all(viewer.layers[0].data == data)
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 3
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 1

    # switch number of displayed dimensions
    viewer.layers[0].data = data[0]
    assert np.all(viewer.layers[0].data == data[0])
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # switch number of displayed dimensions
    viewer.layers[0].data = data[:6]
    assert np.all(viewer.layers[0].data == data[:6])
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 3
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 1

    # change the shape of the data
    viewer.layers[0].data = data[:3]
    assert np.all(viewer.layers[0].data == data[:3])
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 3
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 1


def test_range_one_image(make_test_viewer):
    """Test adding an image with a range one dimensions.

    There should be no slider shown for the axis corresponding to the range
    one dimension.
    """
    np.random.seed(0)
    viewer = make_test_viewer()
    view = viewer.window.qt_viewer

    # add 5D image data with range one dimensions
    data = np.random.random((1, 1, 1, 100, 200))
    viewer.add_image(data)
    assert np.all(viewer.layers[0].data == data)
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 5
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # now add 5D points data - check extra sliders have been created
    points = np.floor(5 * np.random.random((1000, 5))).astype(int)
    points[:, -2:] = 20 * points[:, -2:]
    viewer.add_points(points)
    assert np.all(viewer.layers[1].data == points)
    assert len(viewer.layers) == 2
    assert viewer.dims.ndim == 5
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 3


def test_range_one_images_and_points(make_test_viewer):
    """Test adding images with range one dimensions and points.

    Initially no sliders should be present as the images have range one
    dimensions. On adding the points the sliders should be displayed.
    """
    np.random.seed(0)
    viewer = make_test_viewer()
    view = viewer.window.qt_viewer

    # add 5D image data with range one dimensions
    data = np.random.random((1, 1, 1, 100, 200))
    viewer.add_image(data)
    assert np.all(viewer.layers[0].data == data)
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 5
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # now add 5D points data - check extra sliders have been created
    points = np.floor(5 * np.random.random((1000, 5))).astype(int)
    points[:, -2:] = 20 * points[:, -2:]
    viewer.add_points(points)
    assert np.all(viewer.layers[1].data == points)
    assert len(viewer.layers) == 2
    assert viewer.dims.ndim == 5
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 3


@pytest.mark.filterwarnings("ignore::DeprecationWarning:jupyter_client")
def test_update_console(make_test_viewer):
    """Test updating the console with local variables."""
    viewer = make_test_viewer()
    view = viewer.window.qt_viewer

    # Check viewer in console
    assert view.console.kernel_client is not None
    assert 'viewer' in view.console.shell.user_ns
    assert view.console.shell.user_ns['viewer'] == viewer

    a = 4
    b = 5
    viewer.update_console(locals())
    assert 'a' in view.console.shell.user_ns
    assert view.console.shell.user_ns['a'] == a
    assert 'b' in view.console.shell.user_ns
    assert view.console.shell.user_ns['b'] == b


def test_changing_display_surface(make_test_viewer):
    """Test adding 3D surface and changing its display."""
    viewer = make_test_viewer()
    view = viewer.window.qt_viewer

    np.random.seed(0)
    vertices = 20 * np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    viewer.add_surface(data)
    assert np.all(
        [np.all(vd == d) for vd, d in zip(viewer.layers[0].data, data)]
    )

    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 3
    assert view.dims.nsliders == viewer.dims.ndim

    # Check display is currently 2D with one slider
    assert viewer.layers[0]._data_view.shape[1] == 2
    assert np.sum(view.dims._displayed_sliders) == 1

    # Make display 3D
    viewer.dims.ndisplay = 3
    assert viewer.layers[0]._data_view.shape[1] == 3
    assert np.sum(view.dims._displayed_sliders) == 0

    # Make display 2D again
    viewer.dims.ndisplay = 2
    assert viewer.layers[0]._data_view.shape[1] == 2
    assert np.sum(view.dims._displayed_sliders) == 1

    # Iterate over all values in first dimension
    len_slider = viewer.dims.range[0]
    for s in len_slider:
        viewer.dims.set_point(0, s)


def test_labels_undo_redo(make_test_viewer):
    """Test undoing/redoing on the labels layer."""
    viewer = make_test_viewer()

    data = np.zeros((50, 50), dtype=np.uint8)
    data[:5, :5] = 1
    data[5:10, 5:10] = 2
    data[25:, 25:] = 3

    labels = viewer.add_labels(data)

    l1 = labels.data.copy()

    # fill
    labels.fill((30, 30), 42)

    l2 = labels.data.copy()
    assert not np.array_equal(l1, l2)

    # undo
    labels.undo()
    assert np.array_equal(l1, labels.data)

    # redo
    labels.redo()
    assert np.array_equal(l2, labels.data)

    # history limit
    labels._history_limit = 1
    labels.fill((0, 0), 3)

    l3 = labels.data.copy()

    assert not np.array_equal(l3, l2)

    labels.undo()
    assert np.array_equal(l2, labels.data)

    # cannot undo as limit exceeded
    labels.undo()
    assert np.array_equal(l2, labels.data)
