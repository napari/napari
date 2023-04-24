import os
from unittest.mock import Mock

import numpy as np

from napari._tests.utils import skip_on_win_ci


@skip_on_win_ci
def test_viewer_mouse_bindings(qtbot, make_napari_viewer):
    """Test adding mouse bindings to the viewer"""
    np.random.seed(0)
    viewer = make_napari_viewer()
    view = viewer.window._qt_viewer

    if os.getenv("CI"):
        viewer.show()

    mock_press = Mock()
    mock_drag = Mock()
    mock_release = Mock()
    mock_move = Mock()

    @viewer.mouse_drag_callbacks.append
    def drag_callback(v, event):
        assert viewer == v

        # on press
        mock_press.method()

        yield

        # on move
        while event.type == 'mouse_move':
            mock_drag.method()
            yield

        # on release
        mock_release.method()

    @viewer.mouse_move_callbacks.append
    def move_callback(v, event):
        assert viewer == v
        # on move
        mock_move.method()

    # Simulate press only
    view.canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    mock_press.method.assert_called_once()
    mock_press.reset_mock()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_not_called()

    # Simulate release only
    view.canvas.events.mouse_release(pos=(0, 0), modifiers=(), button=0)
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_called_once()
    mock_release.reset_mock()
    mock_move.method.assert_not_called()

    # Simulate move with no press
    view.canvas.events.mouse_move(pos=(0, 0), modifiers=())
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_called_once()
    mock_move.reset_mock()

    # Simulate press, drag, release
    view.canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    qtbot.wait(10)
    view.canvas.events.mouse_move(
        pos=(0, 0), modifiers=(), button=0, press_event=True
    )
    qtbot.wait(10)
    view.canvas.events.mouse_release(pos=(0, 0), modifiers=(), button=0)
    qtbot.wait(10)
    mock_press.method.assert_called_once()
    mock_drag.method.assert_called_once()
    mock_release.method.assert_called_once()
    mock_move.method.assert_not_called()


@skip_on_win_ci
def test_layer_mouse_bindings(qtbot, make_napari_viewer):
    """Test adding mouse bindings to a layer that is selected"""
    np.random.seed(0)
    viewer = make_napari_viewer()
    view = viewer.window._qt_viewer

    if os.getenv("CI"):
        viewer.show()

    layer = viewer.add_image(np.random.random((10, 20)))
    viewer.layers.selection.add(layer)

    mock_press = Mock()
    mock_drag = Mock()
    mock_release = Mock()
    mock_move = Mock()

    @layer.mouse_drag_callbacks.append
    def drag_callback(_layer, event):
        assert layer == _layer
        # on press
        mock_press.method()

        yield

        # on move
        while event.type == 'mouse_move':
            mock_drag.method()
            yield

        # on release
        mock_release.method()

    @layer.mouse_move_callbacks.append
    def move_callback(_layer, event):
        assert layer == _layer
        # on press
        mock_move.method()

    # Simulate press only
    view.canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    mock_press.method.assert_called_once()
    mock_press.reset_mock()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_not_called()

    # Simulate release only
    view.canvas.events.mouse_release(pos=(0, 0), modifiers=(), button=0)
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_called_once()
    mock_release.reset_mock()
    mock_move.method.assert_not_called()

    # Simulate move with no press
    view.canvas.events.mouse_move(pos=(0, 0), modifiers=())
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_called_once()
    mock_move.reset_mock()

    # Simulate press, drag, release
    view.canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    qtbot.wait(10)
    view.canvas.events.mouse_move(
        pos=(0, 0), modifiers=(), button=0, press_event=True
    )
    qtbot.wait(10)
    view.canvas.events.mouse_release(pos=(0, 0), modifiers=(), button=0)
    qtbot.wait(10)
    mock_press.method.assert_called_once()
    mock_drag.method.assert_called_once()
    mock_release.method.assert_called_once()
    mock_move.method.assert_not_called()


@skip_on_win_ci
def test_unselected_layer_mouse_bindings(qtbot, make_napari_viewer):
    """Test adding mouse bindings to a layer that is not selected"""
    np.random.seed(0)
    viewer = make_napari_viewer()
    view = viewer.window._qt_viewer

    if os.getenv("CI"):
        viewer.show()

    layer = viewer.add_image(np.random.random((10, 20)))
    viewer.layers.selection.remove(layer)

    mock_press = Mock()
    mock_drag = Mock()
    mock_release = Mock()
    mock_move = Mock()

    @layer.mouse_drag_callbacks.append
    def drag_callback(_layer, event):
        assert layer == _layer
        # on press
        mock_press.method()

        yield

        # on move
        while event.type == 'mouse_move':
            mock_drag.method()
            yield

        # on release
        mock_release.method()

    @layer.mouse_move_callbacks.append
    def move_callback(_layer, event):
        assert layer == _layer
        # on press
        mock_move.method()

    # Simulate press only
    view.canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_not_called()

    # Simulate release only
    view.canvas.events.mouse_release(pos=(0, 0), modifiers=(), button=0)
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_not_called()

    # Simulate move with no press
    view.canvas.events.mouse_move(pos=(0, 0), modifiers=())
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_not_called()

    # Simulate press, drag, release
    view.canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    qtbot.wait(10)
    view.canvas.events.mouse_move(
        pos=(0, 0), modifiers=(), button=0, press_event=True
    )
    qtbot.wait(10)
    view.canvas.events.mouse_release(pos=(0, 0), modifiers=(), button=0)
    qtbot.wait(10)
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_not_called()
