from unittest import mock

import pytest

from napari._app_model import get_app


@pytest.mark.parametrize('perfmon_value', [True, False])
@pytest.mark.filterwarnings(
    'ignore:Using NAPARI_PERFMON with an already-running QtApp'
)  # See napari/napari#6957
def test_menu_visibility(monkeypatch, make_napari_viewer, perfmon_value):
    """Test debug menu visibility following performance monitor usage."""
    monkeypatch.setattr('napari.utils.perf.USE_PERFMON', perfmon_value)
    viewer = make_napari_viewer()
    # Check the menu is available or not following `NAPARI_PERFMON` value
    #   * `NAPARI_PERFMON=1` -> `perf.USE_PERFMON==True` -> Debug menu available
    #   * `NAPARI_PERFMON=` -> `perf.USE_PERFMON==False` -> Debug menu shouldn't exist
    assert bool(getattr(viewer.window, '_debug_menu', None)) == perfmon_value

    # Stop perf widget timer to prevent test failure on teardown when needed
    if perfmon_value:
        viewer.window._qt_viewer.dockPerformance.widget().timer.stop()


@pytest.mark.filterwarnings(
    'ignore:Using NAPARI_PERFMON with an already-running QtApp'
)  # See napari/napari#6957
def test_start_trace_actions(monkeypatch, make_napari_viewer):
    """Test triggering the start trace action shows dialog to select trace file"""
    monkeypatch.setattr('napari.utils.perf.USE_PERFMON', True)
    app = get_app()
    viewer = make_napari_viewer()

    # Check action execution calls a dialog to get name file to use
    with mock.patch(
        'napari._qt._qapp_model.qactions._debug.QFileDialog'
    ) as mock_dialog:
        mock_dialog_instance = mock_dialog.return_value
        mock_save = mock_dialog_instance.getSaveFileName
        mock_save.return_value = (None, None)
        app.commands.execute_command('napari.window.debug.start_trace_dialog')
    mock_dialog.assert_called_once()
    mock_save.assert_called_once()

    # Stop perf widget timer to prevent test failure on teardown
    viewer.window._qt_viewer.dockPerformance.widget().timer.stop()
