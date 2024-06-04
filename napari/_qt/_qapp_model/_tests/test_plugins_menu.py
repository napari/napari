from unittest import mock

import pytest

from napari._app_model import get_app
from napari._qt._qapp_model.qactions._plugins import (
    _plugin_manager_dialog_avail,
)


@pytest.mark.skipif(
    not _plugin_manager_dialog_avail(),
    reason='`napari_plugin_manager` not available',
)
def test_plugin_manager_action(make_napari_viewer):
    """
    Test manage plugins installation action.

    The test is skipped in case `napari_plugin_manager` is not available
    """
    app = get_app()
    viewer = make_napari_viewer()

    with mock.patch(
        'napari_plugin_manager.qt_plugin_dialog.QtPluginDialog'
    ) as mock_plugin_dialog:
        app.commands.execute_command(
            'napari.window.plugins.plugin_install_dialog'
        )
    mock_plugin_dialog.assert_called_once_with(viewer.window._qt_window)


def test_plugin_errors_action(make_napari_viewer):
    """Test plugin errors action."""
    make_napari_viewer()
    app = get_app()

    with mock.patch(
        'napari._qt._qapp_model.qactions._plugins.QtPluginErrReporter.exec_'
    ) as mock_plugin_dialog:
        app.commands.execute_command(
            'napari.window.plugins.plugin_err_reporter'
        )
    mock_plugin_dialog.assert_called_once()
