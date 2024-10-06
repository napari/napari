import importlib
from unittest import mock

import pytest

from napari._app_model import get_app_model
from napari._qt._qapp_model._tests.utils import get_submenu_action


@pytest.fixture(params=[True, False])
def perfmon_activation(monkeypatch, request):
    if request.param:
        # If perfmon needs to be active add env var and reload modules
        monkeypatch.setenv('NAPARI_PERFMON', '1')
        from napari.utils import perf

        importlib.reload(perf._config)
        importlib.reload(perf._timers)
        importlib.reload(perf)

        # Check `USE_PERFMON` values correspond to env var state
        assert perf.perf_config is not None
        assert perf._timers.USE_PERFMON
        assert perf.USE_PERFMON
    else:
        # If perfmon doesn't need to be active remove env var and reload modules
        monkeypatch.delenv('NAPARI_PERFMON', raising=False)
        from napari.utils import perf

        importlib.reload(perf._config)
        importlib.reload(perf._timers)
        importlib.reload(perf)

        # Check `USE_PERFMON` values correspond to env var state
        assert perf.perf_config is None
        assert not perf._timers.USE_PERFMON
        assert not perf.USE_PERFMON

    yield request.param

    # On teardown always try to remove env var and reload `perf` module
    monkeypatch.delenv('NAPARI_PERFMON', raising=False)
    from napari.utils import perf

    importlib.reload(perf._config)
    importlib.reload(perf._timers)
    importlib.reload(perf)

    # Check `USE_PERFMON` values correspond to env var state
    assert perf.perf_config is None
    assert not perf._timers.USE_PERFMON
    assert not perf.USE_PERFMON


@pytest.mark.filterwarnings(
    'ignore:Using NAPARI_PERFMON with an already-running QtApp'
)  # TODO: remove once napari/napari#6957 resolved
def test_debug_menu_exists(perfmon_activation, make_napari_viewer, qtbot):
    """Test debug menu existence following performance monitor usage."""
    use_perfmon = perfmon_activation
    viewer = make_napari_viewer()

    # Check the menu exists following `NAPARI_PERFMON` value
    #   * `NAPARI_PERFMON=1` -> `perf.USE_PERFMON==True` -> Debug menu available
    #   * `NAPARI_PERFMON=` -> `perf.USE_PERFMON==False` -> Debug menu shouldn't exist
    assert bool(getattr(viewer.window, '_debug_menu', None)) == use_perfmon

    # Stop perf widget timer to prevent test failure on teardown when needed
    if use_perfmon:
        viewer.window._qt_viewer.dockPerformance.widget().timer.stop()


@pytest.mark.filterwarnings(
    'ignore:Using NAPARI_PERFMON with an already-running QtApp'
)  # TODO: remove once napari/napari#6957 resolved
def test_start_stop_trace_actions(
    perfmon_activation, make_napari_viewer, tmp_path, qtbot
):
    """Test start and stop recording trace actions."""
    use_perfmon = perfmon_activation
    if use_perfmon:
        trace_file = tmp_path / 'trace.json'
        app = get_app_model()
        viewer = make_napari_viewer()

        # Check Debug menu exists and actions state
        assert getattr(viewer.window, '_debug_menu', None) is not None

        start_action, menu = get_submenu_action(
            viewer.window._debug_menu,
            'Performance Trace',
            'Start Recording...',
        )
        stop_action, menu = get_submenu_action(
            viewer.window._debug_menu, 'Performance Trace', 'Stop Recording...'
        )

        # Check initial actions state
        viewer.window._debug_menu.aboutToShow.emit()
        assert start_action.isEnabled()
        assert not stop_action.isEnabled()

        # Check start action execution
        def assert_start_recording():
            viewer.window._debug_menu.aboutToShow.emit()
            assert not start_action.isEnabled()
            assert stop_action.isEnabled()

        with mock.patch(
            'napari._qt._qapp_model.qactions._debug.QFileDialog'
        ) as mock_dialog:
            mock_dialog_instance = mock_dialog.return_value
            mock_save = mock_dialog_instance.getSaveFileName
            mock_save.return_value = (str(trace_file), None)
            app.commands.execute_command(
                'napari.window.debug.start_trace_dialog'
            )
        mock_dialog.assert_called_once()
        mock_save.assert_called_once()
        assert not trace_file.exists()
        qtbot.waitUntil(assert_start_recording)

        # Check stop action execution
        def assert_stop_recording():
            viewer.window._debug_menu.aboutToShow.emit()
            assert start_action.isEnabled()
            assert not stop_action.isEnabled()
            assert trace_file.exists()

        app.commands.execute_command('napari.window.debug.stop_trace')
        qtbot.waitUntil(assert_stop_recording)

        # Stop perf widget timer to prevent test failure on teardown
        viewer.window._qt_viewer.dockPerformance.widget().timer.stop()
        qtbot.waitUntil(
            lambda: not viewer.window._qt_viewer.dockPerformance.widget().timer.isActive()
        )
    else:
        # Nothing to test
        pytest.skip('Perfmon is disabled')
