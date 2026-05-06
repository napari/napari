from unittest.mock import patch

import numpy as np


def test_data_change_ndisplay_image(make_napari_viewer):
    """Test change data calls for image layer with ndisplay change."""
    viewer = make_napari_viewer()

    np.random.seed(0)
    data = np.random.random((10, 15, 8))
    layer = viewer.add_image(data)
    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]

    @patch.object(visual, '_on_data_change', wraps=visual._on_data_change)
    def test_ndisplay_change(mocked_method, ndisplay=3):
        viewer.dims.ndisplay = ndisplay
        mocked_method.assert_called_once()

    # Switch to 3D rendering mode and back to 2D rendering mode
    test_ndisplay_change(ndisplay=3)
    test_ndisplay_change(ndisplay=2)


def test_data_change_ndisplay_labels(make_napari_viewer):
    """Test change data calls for labels layer with ndisplay change."""
    viewer = make_napari_viewer()

    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15, 8))
    layer = viewer.add_labels(data)

    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]

    @patch.object(visual, '_on_data_change', wraps=visual._on_data_change)
    def test_ndisplay_change(mocked_method, ndisplay=3):
        viewer.dims.ndisplay = ndisplay
        mocked_method.assert_called_once()

    # Switch to 3D rendering mode and back to 2D rendering mode
    test_ndisplay_change(ndisplay=3)
    test_ndisplay_change(ndisplay=2)


def test_data_change_ndisplay_points(make_napari_viewer):
    """Test change data calls for points layer with ndisplay change."""
    viewer = make_napari_viewer()

    np.random.seed(0)
    data = 20 * np.random.random((10, 3))
    layer = viewer.add_points(data)
    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]

    @patch.object(visual, '_on_data_change', wraps=visual._on_data_change)
    def test_ndisplay_change(mocked_method, ndisplay=3):
        viewer.dims.ndisplay = ndisplay
        mocked_method.assert_called_once()

    # Switch to 3D rendering mode and back to 2D rendering mode
    test_ndisplay_change(ndisplay=3)
    test_ndisplay_change(ndisplay=2)


def test_data_change_ndisplay_vectors(make_napari_viewer):
    """Test change data calls for vectors layer with ndisplay change."""
    viewer = make_napari_viewer()

    np.random.seed(0)
    data = 20 * np.random.random((10, 2, 3))
    layer = viewer.add_vectors(data)
    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]

    @patch.object(visual, '_on_data_change', wraps=visual._on_data_change)
    def test_ndisplay_change(mocked_method, ndisplay=3):
        viewer.dims.ndisplay = ndisplay
        mocked_method.assert_called_once()

    # Switch to 3D rendering mode and back to 2D rendering mode
    test_ndisplay_change(ndisplay=3)
    test_ndisplay_change(ndisplay=2)


def test_data_change_ndisplay_shapes(make_napari_viewer):
    """Test change data calls for shapes layer with ndisplay change."""
    viewer = make_napari_viewer()

    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 3))
    layer = viewer.add_shapes(data)

    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]

    @patch.object(visual, '_on_data_change', wraps=visual._on_data_change)
    def test_ndisplay_change(mocked_method, ndisplay=3):
        viewer.dims.ndisplay = ndisplay
        mocked_method.assert_called_once()

    # Switch to 3D rendering mode and back to 2D rendering mode
    test_ndisplay_change(ndisplay=3)
    test_ndisplay_change(ndisplay=2)


def test_data_change_ndisplay_surface(make_napari_viewer):
    """Test change data calls for surface layer with ndisplay change."""
    viewer = make_napari_viewer()

    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = viewer.add_surface(data)

    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]

    @patch.object(visual, '_on_data_change', wraps=visual._on_data_change)
    def test_ndisplay_change(mocked_method, ndisplay=3):
        viewer.dims.ndisplay = ndisplay
        mocked_method.assert_called_once()

    # Switch to 3D rendering mode and back to 2D rendering mode
    test_ndisplay_change(ndisplay=3)
    test_ndisplay_change(ndisplay=2)


def test_add_invisible_image_layer_in_3d(make_napari_viewer):
    """Adding an Image with visible=False while the viewer is already in 3D
    must not crash vispy. Without the ScalarFieldSlicingState fix, the empty
    placeholder cached at layer init has rank 2 (ndisplay defaults to 2),
    set_view_slice is skipped on add because the layer is invisible, and
    vispy then rejects the wrong-rank array with
    "Volume visual needs a 3D array.".
    """
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = viewer.add_image(np.zeros((4, 5, 6), dtype=np.uint8), visible=False)
    assert layer.visible is False
    # Toggling visibility on then off exercises the slice-recompute path now
    # that the placeholder is at the right rank.
    layer.visible = True
    layer.visible = False


def test_add_invisible_labels_layer_in_3d(make_napari_viewer):
    """Same regression as test_add_invisible_image_layer_in_3d, but for
    Labels. Both layer types share ScalarFieldSlicingState so the bug and
    fix apply to both.
    """
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = viewer.add_labels(
        np.zeros((4, 5, 6), dtype=np.uint8), visible=False
    )
    assert layer.visible is False
    layer.visible = True
    layer.visible = False
