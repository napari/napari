import sys
from unittest import mock

import pytest

from napari._app_model import get_app


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
    app = get_app()
    viewer = make_napari_viewer()

    with mock.patch(
        'napari._qt.dialogs.qt_about.QtAbout.showAbout'
    ) as mock_about:
        app.commands.execute_command(action_id)
    mock_about.assert_called_once_with(viewer.window._qt_window)
