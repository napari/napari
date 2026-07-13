"""For testing the Help menu"""

import sys
from unittest import mock

import pytest
import requests
from qtpy.QtCore import Qt

from napari._app_model import get_app_model
from napari._qt._qapp_model.qactions._help import HELP_URLS


@pytest.mark.parametrize('url', HELP_URLS.keys())
def test_help_urls(url):
    if url == 'release_notes':
        pytest.skip('No release notes for dev version')

    r = requests.head(HELP_URLS[url])
    r.raise_for_status()


@pytest.mark.parametrize(
    'action_id',
    [
        'napari.window.help.info',
        'napari.window.help.about_macos',
    ]
    if sys.platform == 'darwin'
    else ['napari.window.help.info'],
)
def test_about_action(make_napari_viewer, action_id):
    app = get_app_model()
    viewer = make_napari_viewer()

    with mock.patch(
        'napari._qt.dialogs.qt_about.QtAbout.showAbout'
    ) as mock_about:
        app.commands.execute_command(action_id)
    mock_about.assert_called_once_with(viewer.window._qt_window)


def test_take_a_tour_action(make_napari_viewer, qtbot):
    app = get_app_model()
    viewer = make_napari_viewer(show=True)

    app.commands.execute_command('napari.window.help.viewer_tour')

    qtbot.waitUntil(lambda: viewer.window._qt_window._viewer_tour is not None)
    assert len(viewer.layers) == 1
    assert viewer.window._qt_window._viewer_tour._tooltip.isVisible()

    tour = viewer.window._qt_window._viewer_tour
    app.commands.execute_command('napari.window.help.viewer_tour')
    assert viewer.window._qt_window._viewer_tour is tour

    assert tour._current == 0
    qtbot.keyClick(viewer.window._qt_window, Qt.Key.Key_N)
    assert tour._current == 1
    qtbot.keyClick(viewer.window._qt_window, Qt.Key.Key_P)
    assert tour._current == 0
    qtbot.keyClick(viewer.window._qt_window, Qt.Key.Key_Escape)
    qtbot.waitUntil(lambda: viewer.window._qt_window._viewer_tour is None)
