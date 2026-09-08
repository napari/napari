from unittest import mock

import pytest


@pytest.mark.parametrize('stack', [True, False])
def test_open_files_dialog(make_napari_viewer, stack):
    """Check `QtViewer._open_files_dialog` correnct when `stack=True`."""
    viewer = make_napari_viewer()
    with (
        mock.patch(
            'napari._qt.qt_viewer.QtViewer._open_file_dialog_uni'
        ) as mock_file,
        mock.patch('napari._qt.qt_viewer.QtViewer._qt_open') as mock_open,
    ):
        viewer.window._qt_viewer._open_files_dialog(stack=stack)
    mock_open.assert_called_once_with(
        mock_file.return_value,
        choose_plugin=False,
        stack=stack,
    )
