from unittest.mock import Mock

import numpy as np
from vispy import keys


def test_viewer_key_bindings(make_napari_viewer):
    """Test adding key bindings to the viewer"""
    np.random.seed(0)
    viewer = make_napari_viewer()
    view = viewer.window._qt_viewer.canvas

    mock_press = Mock()
    mock_release = Mock()
    mock_shift_press = Mock()
    mock_shift_release = Mock()

    @viewer.bind_key('F')
    def key_callback(v):
        assert viewer == v

        # on press
        mock_press.method()

        yield

        # on release
        mock_release.method()

    @viewer.bind_key('Shift-F')
    def key_shift_callback(v):
        assert viewer == v

        # on press
        mock_shift_press.method()

        yield

        # on release
        mock_shift_release.method()

    # Simulate press only
    view.scene_canvas.events.key_press(key=keys.Key('F'))
    mock_press.method.assert_called_once()
    mock_press.reset_mock()
    mock_release.method.assert_not_called()
    mock_shift_press.method.assert_not_called()
    mock_shift_release.method.assert_not_called()

    # Simulate release only
    view.scene_canvas.events.key_release(key=keys.Key('F'))
    mock_press.method.assert_not_called()
    mock_release.method.assert_called_once()
    mock_release.reset_mock()
    mock_shift_press.method.assert_not_called()
    mock_shift_release.method.assert_not_called()

    # Simulate press only
    view.scene_canvas.events.key_press(
        key=keys.Key('F'), modifiers=[keys.SHIFT]
    )
    mock_press.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_shift_press.method.assert_called_once()
    mock_shift_press.reset_mock()
    mock_shift_release.method.assert_not_called()

    # Simulate release only
    view.scene_canvas.events.key_release(
        key=keys.Key('F'), modifiers=[keys.SHIFT]
    )
    mock_press.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_shift_press.method.assert_not_called()
    mock_shift_release.method.assert_called_once()
    mock_shift_release.reset_mock()


def test_layer_key_bindings(make_napari_viewer):
    """Test adding key bindings to a layer"""
    np.random.seed(0)
    viewer = make_napari_viewer()
    view = viewer.window._qt_viewer.canvas

    layer = viewer.add_image(np.random.random((10, 20)))
    viewer.layers.selection.add(layer)

    mock_press = Mock()
    mock_release = Mock()
    mock_shift_press = Mock()
    mock_shift_release = Mock()

    @layer.bind_key('F')
    def key_callback(_layer):
        assert layer == _layer
        # on press
        mock_press.method()
        yield
        # on release
        mock_release.method()

    @layer.bind_key('Shift-F')
    def key_shift_callback(_layer):
        assert layer == _layer

        # on press
        mock_shift_press.method()

        yield

        # on release
        mock_shift_release.method()

    # Simulate press only
    view.scene_canvas.events.key_press(key=keys.Key('F'))
    mock_press.method.assert_called_once()
    mock_press.reset_mock()
    mock_release.method.assert_not_called()
    mock_shift_press.method.assert_not_called()
    mock_shift_release.method.assert_not_called()

    # Simulate release only
    view.scene_canvas.events.key_release(key=keys.Key('F'))
    mock_press.method.assert_not_called()
    mock_release.method.assert_called_once()
    mock_release.reset_mock()
    mock_shift_press.method.assert_not_called()
    mock_shift_release.method.assert_not_called()

    # Simulate press only
    view.scene_canvas.events.key_press(
        key=keys.Key('F'), modifiers=[keys.SHIFT]
    )
    mock_press.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_shift_press.method.assert_called_once()
    mock_shift_press.reset_mock()
    mock_shift_release.method.assert_not_called()

    # Simulate release only
    view.scene_canvas.events.key_release(
        key=keys.Key('F'), modifiers=[keys.SHIFT]
    )
    mock_press.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_shift_press.method.assert_not_called()
    mock_shift_release.method.assert_called_once()
    mock_shift_release.reset_mock()


def test_reset_scroll_progress(make_napari_viewer):
    """Test select all key binding."""
    viewer = make_napari_viewer()
    view = viewer.window._qt_viewer.canvas
    assert viewer.dims._scroll_progress == 0

    view.scene_canvas.events.key_press(key=keys.Key('Control'))
    viewer.dims._scroll_progress = 10
    assert viewer.dims._scroll_progress == 10

    view.scene_canvas.events.key_release(key=keys.Key('Control'))
    assert viewer.dims._scroll_progress == 0
