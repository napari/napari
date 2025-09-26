"""Tests of the Viewer class that interact directly with the Qt code"""

from __future__ import annotations

import os
import textwrap
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy import testing as npt
from qtpy.QtCore import QEvent, Qt, QUrl
from qtpy.QtGui import QGuiApplication, QKeyEvent
from qtpy.QtWidgets import QApplication

from napari._qt._tests.test_qt_viewer import qt_viewer
from napari._tests.utils import skip_local_popups, skip_on_win_ci
from napari.settings import get_settings
from napari.utils.theme import available_themes


@pytest.mark.usefixtures('builtins')
@pytest.mark.enable_console
@pytest.mark.filterwarnings('ignore::DeprecationWarning:jupyter_client')
@patch('qtpy.QtWidgets.QApplication.topLevelWidgets')
def test_drop_python_file(qapp_mock, make_napari_viewer, tmp_path):
    """Test dropping a python file on to the viewer."""
    viewer = make_napari_viewer()
    filename = tmp_path / 'image_to_drop.py'
    file_content = textwrap.dedent("""
    import numpy as np
    import napari

    data = np.zeros((10, 10))
    viewer = napari.Viewer()
    viewer.add_image(data, name='Dropped Image')

    napari.run()
    """)
    filename.write_text(file_content)

    # Simulate dropping the file
    mock_event = MagicMock()
    mock_event.mimeData.return_value.urls.return_value = [
        QUrl(filename.as_uri())
    ]
    viewer.window._qt_viewer.dropEvent(mock_event)

    # Check that the file was executed
    assert viewer.layers[0].name == 'Dropped Image'

    # Ensure that napari.run is stopped early and event loops are not started
    # by call napari.run() in the script.
    qapp_mock.assert_not_called()

    # Check that the console is updated with locals from the script
    console = viewer.window._qt_viewer.console
    assert 'data' in console.shell.user_ns
    assert np.array_equal(console.shell.user_ns['data'], np.zeros((10, 10)))


@pytest.mark.usefixtures('builtins')
@pytest.mark.enable_console
@pytest.mark.filterwarnings('ignore::DeprecationWarning:jupyter_client')
def test_drop_python_file_3d(make_napari_viewer, tmp_path):
    """Test that dropping a python file using a 3D image on the viewer works."""
    viewer = make_napari_viewer()
    filename = tmp_path / 'image_to_drop_3d.py'
    file_content = textwrap.dedent("""
    import numpy as np
    from napari import Viewer

    data = np.zeros((2, 10, 10))
    viewer = Viewer(ndisplay=3)
    viewer.add_image(data, name='Dropped Image')
    """)
    filename.write_text(file_content)

    # Simulate dropping the file
    mock_event = MagicMock()
    mock_event.mimeData.return_value.urls.return_value = [
        QUrl(filename.as_uri())
    ]
    viewer.window._qt_viewer.dropEvent(mock_event)

    # Check that the file was executed
    assert viewer.layers[0].name == 'Dropped Image'
    assert viewer.dims.ndim == 3

    # Check that the console is updated with locals from the script
    console = viewer.window._qt_viewer.console
    assert 'data' in console.shell.user_ns
    assert np.array_equal(console.shell.user_ns['data'], np.zeros((2, 10, 10)))


@pytest.mark.usefixtures('builtins')
@pytest.mark.enable_console
@pytest.mark.filterwarnings('ignore::DeprecationWarning:jupyter_client')
def test_drop_python_file_double_viewer(make_napari_viewer, tmp_path):
    """Test that dropping a python file on the viewer works."""
    viewer = make_napari_viewer()
    filename = tmp_path / 'test.py'
    file_content = textwrap.dedent("""
    import numpy as np
    from napari import Viewer

    data = np.zeros((10, 10))
    viewer1 = Viewer()
    viewer1.add_image(data, name='Dropped Image')
    viewer2 = Viewer(title="text")
    viewer2.add_image(data, name='Dropped Image 2')
    """)
    filename.write_text(file_content)

    # Simulate dropping the file
    mock_event = MagicMock()
    mock_event.mimeData.return_value.urls.return_value = [
        QUrl(filename.as_uri())
    ]
    viewer.window._qt_viewer.dropEvent(mock_event)

    # Check that the file was executed
    assert viewer.layers[0].name == 'Dropped Image'
    assert len(viewer._instances) == 2  # Two viewers should be created
    instances = list(viewer._instances)
    idx = 0 if instances[1] == viewer else 1
    assert instances[idx].title == 'text'  # Check the second viewer's name
    instances[idx].close()  # Close the second viewer

    # check that the console is updated with locals from the script
    console = viewer.window._qt_viewer.console
    assert 'data' in console.shell.user_ns
    assert np.array_equal(console.shell.user_ns['data'], np.zeros((10, 10)))
    assert 'viewer1' in console.shell.user_ns
    assert 'viewer2' in console.shell.user_ns


def test_qt_viewer(make_napari_viewer):
    """Test instantiating viewer."""
    viewer = make_napari_viewer()
    view = viewer.window._qt_viewer

    assert viewer.title == 'napari'
    assert view.viewer == viewer

    assert len(viewer.layers) == 0
    assert view.layers.model().rowCount() == 0

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_qt_viewer_with_console(make_napari_viewer):
    """Test instantiating console from viewer."""
    viewer = make_napari_viewer()
    view = viewer.window._qt_viewer
    # Check console is created when requested
    assert view.console is not None
    assert view.dockConsole.widget() is view.console


def test_qt_viewer_toggle_console(make_napari_viewer):
    """Test instantiating console from viewer."""
    viewer = make_napari_viewer()
    view = viewer.window._qt_viewer
    # Check console has been created when it is supposed to be shown
    view.toggle_console_visibility(None)
    assert view._console is not None
    assert view.dockConsole.widget() is view.console


@skip_local_popups
@pytest.mark.skipif(os.environ.get('MIN_REQ', '0') == '1', reason='min req')
def test_qt_viewer_console_focus(qtbot, make_napari_viewer):
    """Test console has focus when instantiating from viewer."""
    viewer = make_napari_viewer(show=True)
    view = viewer.window._qt_viewer
    assert not view.console.hasFocus(), 'console has focus before being shown'

    view.toggle_console_visibility(None)

    def console_has_focus():
        assert view.console.hasFocus(), (
            'console does not have focus when shown'
        )

    qtbot.waitUntil(console_has_focus)


@skip_on_win_ci
def test_screenshot(make_napari_viewer, qapp, qtbot):
    "Test taking a screenshot"
    viewer = make_napari_viewer(show=True)
    rng = np.random.default_rng(0)
    # Add image
    data = rng.random((10, 15))
    viewer.add_image(data)

    # Add labels
    data = rng.integers(20, size=(10, 15))
    viewer.add_labels(data)

    # Add points
    data = 20 * rng.random((10, 2))
    viewer.add_points(data)

    # Add vectors
    data = 20 * rng.random((10, 2, 2))
    viewer.add_vectors(data)

    # Add shapes
    data = 20 * rng.random((10, 4, 2))
    viewer.add_shapes(data)
    # wait for window be fully rendered
    qtbot.wait_exposed(qt_viewer)

    # without these two lines, the shape of screenshot1 and screenshot2 differ by
    # 2 pixels. It looks like some event requires inactivity in event loop to
    # trigger resize to final shape.
    qtbot.wait(5)
    qapp.processEvents()

    screenshot2 = viewer.window.screenshot(flash=False, canvas_only=True)

    qapp.processEvents()

    # Take screenshot
    with pytest.warns(FutureWarning, match='qt_viewer'):
        screenshot1 = viewer.window.qt_viewer.screenshot(flash=False)

    npt.assert_array_equal(screenshot1, screenshot2)
    assert screenshot1.ndim == 3


@pytest.mark.slow
@skip_on_win_ci
def test_qt_viewer_clipboard_with_flash(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    # make sure clipboard is empty
    QGuiApplication.clipboard().clear()
    clipboard_image = QGuiApplication.clipboard().image()
    assert clipboard_image.isNull()

    viewer.window._qt_viewer.clipboard(flash=True)

    viewer.window.clipboard(flash=False, canvas_only=True)

    clipboard_image = QGuiApplication.clipboard().image()
    assert not clipboard_image.isNull()

    # ensure the flash effect is applied
    assert (
        viewer.window._qt_viewer._welcome_widget.graphicsEffect() is not None
    )
    assert hasattr(
        viewer.window._qt_viewer._welcome_widget, '_flash_animation'
    )
    qtbot.wait(500)  # wait for the animation to finish
    assert viewer.window._qt_viewer._welcome_widget.graphicsEffect() is None
    assert not hasattr(
        viewer.window._qt_viewer._welcome_widget, '_flash_animation'
    )

    # clear clipboard and grab image from application view
    QGuiApplication.clipboard().clear()
    clipboard_image = QGuiApplication.clipboard().image()
    assert clipboard_image.isNull()

    # capture screenshot of the entire window
    viewer.window.clipboard(flash=True)
    clipboard_image = QGuiApplication.clipboard().image()
    assert not clipboard_image.isNull()

    # ensure the flash effect is applied
    assert viewer.window._qt_window.graphicsEffect() is not None
    assert hasattr(viewer.window._qt_window, '_flash_animation')
    qtbot.wait(500)  # wait for the animation to finish
    assert viewer.window._qt_window.graphicsEffect() is None
    assert not hasattr(viewer.window._qt_window, '_flash_animation')


@skip_on_win_ci
def test_qt_viewer_clipboard_without_flash(make_napari_viewer):
    viewer = make_napari_viewer()
    # make sure clipboard is empty
    QGuiApplication.clipboard().clear()
    clipboard_image = QGuiApplication.clipboard().image()
    assert clipboard_image.isNull()

    # capture screenshot
    viewer.window._qt_viewer.clipboard(flash=False)

    viewer.window.clipboard(flash=False, canvas_only=True)

    clipboard_image = QGuiApplication.clipboard().image()
    assert not clipboard_image.isNull()

    # ensure the flash effect is not applied
    assert viewer.window._qt_viewer._welcome_widget.graphicsEffect() is None
    assert not hasattr(
        viewer.window._qt_viewer._welcome_widget, '_flash_animation'
    )

    # clear clipboard and grab image from application view
    QGuiApplication.clipboard().clear()
    clipboard_image = QGuiApplication.clipboard().image()
    assert clipboard_image.isNull()

    # capture screenshot of the entire window
    viewer.window.clipboard(flash=False)
    clipboard_image = QGuiApplication.clipboard().image()
    assert not clipboard_image.isNull()

    # ensure the flash effect is not applied
    assert viewer.window._qt_window.graphicsEffect() is None
    assert not hasattr(viewer.window._qt_window, '_flash_animation')


@pytest.mark.parametrize('theme', available_themes())
def test_canvas_color(make_napari_viewer, theme):
    """Test instantiating viewer with different themes.

    See: https://github.com/napari/napari/issues/3278
    """
    # This test is to make sure the application starts with
    # with different themes
    get_settings().appearance.theme = theme
    viewer = make_napari_viewer()
    assert viewer.theme == theme


def test_shortcut_passing(make_napari_viewer):
    viewer = make_napari_viewer(ndisplay=3)
    layer = viewer.add_labels(
        np.zeros((2, 2, 2), dtype=np.uint8), scale=(1, 2, 4)
    )
    layer.mode = 'fill'

    qt_window = viewer.window._qt_window

    qt_window.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier
        )
    )
    qt_window.keyReleaseEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier
        )
    )
    assert layer.mode == 'erase'


def test_points_2d_to_3d(make_napari_viewer):
    """See https://github.com/napari/napari/issues/6925"""
    # this requires a full viewer cause some issues are caused only by
    # qt processing events
    viewer = make_napari_viewer(ndisplay=2, show=True)
    viewer.add_points()
    QApplication.processEvents()
    viewer.dims.ndisplay = 3
    QApplication.processEvents()


def test_surface_mixed_dim(make_napari_viewer):
    """Test that adding a layer that changes the world ndim
    when ndisplay=3 before the mouse cursor has been updated
    doesn't raise an error.

    See PR: https://github.com/napari/napari/pull/3881
    """
    viewer = make_napari_viewer(ndisplay=3)

    verts = np.array([[0, 0, 0], [0, 20, 10], [10, 0, -10], [10, 10, -10]])
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    values = np.linspace(0, 1, len(verts))
    data = (verts, faces, values)
    viewer.add_surface(data)

    timeseries_values = np.vstack([values, values])
    timeseries_data = (verts, faces, timeseries_values)
    viewer.add_surface(timeseries_data)
