from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from qtpy.QtCore import QFileSystemWatcher
from qtpy.QtWidgets import QDockWidget

from napari._hot_reload import _require_qtreload
from napari._qt._qapp_model.qactions._help import _show_tips
from napari._qt.qt_resources import STYLES
from napari._qt.widgets.qt_tips import TipsWidget

if TYPE_CHECKING:
    from pathlib import Path

    from pytestqt.qtbot import QtBot

    from napari.utils._testsupport import ViewerFactory


@pytest.fixture
def hot_reload_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        _require_qtreload()
    except ImportError as e:
        pytest.skip(str(e))
    monkeypatch.setenv('NAPARI_DEV', '1')
    monkeypatch.delenv('NAPARI_DEV_MODULES', raising=False)


def test_hot_reload_is_opt_in(
    monkeypatch: pytest.MonkeyPatch, make_napari_viewer: ViewerFactory
) -> None:
    monkeypatch.delenv('NAPARI_DEV', raising=False)
    window = make_napari_viewer().window
    assert window._qt_window.findChild(QDockWidget, 'QDev') is None


def test_hot_reload_env_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
    make_napari_viewer: ViewerFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv('NAPARI_DEV', '1')
    with patch('napari._hot_reload.version', side_effect=PackageNotFoundError):
        window = make_napari_viewer().window

    assert window._qt_window.findChild(QDockWidget, 'QDev') is None
    assert 'napari[reload]' in caplog.text


@pytest.mark.usefixtures('hot_reload_enabled')
def test_hot_reload_dock_and_inspector_cleanup(
    make_napari_viewer: ViewerFactory, qtbot: QtBot
) -> None:
    viewer = make_napari_viewer(show=True)
    window = viewer.window
    widget = window._hot_reload_widget
    assert widget is not None
    dock = window._qt_window.findChild(QDockWidget, 'QDev')
    assert dock is not None
    assert dock.widget() is widget
    assert dock.toggleViewAction() in window.window_menu.actions()
    assert widget.reloader.modules == ('napari', 'napari_builtins')

    widget.inspector.start()
    assert widget.inspector.active
    dock.hide()
    assert not widget.inspector.active

    dock.show()
    widget.inspector.start()
    watcher = widget.reloader.findChild(QFileSystemWatcher)
    assert watcher is not None
    assert watcher.files()
    with (
        qtbot.waitSignal(watcher.destroyed),
        qtbot.waitSignal(widget.inspector.evt_active_changed),
    ):
        viewer.close()
    assert window._hot_reload_widget is None


@pytest.mark.usefixtures('hot_reload_enabled')
def test_hot_reload_updates_stylesheet(
    make_napari_viewer: ViewerFactory,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / 'napari_test_styles'
    package.mkdir()
    (package / '__init__.py').touch()
    stylesheet = package / 'test.qss'
    stylesheet.write_text('QWidget { color: #123456; }', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(STYLES, 'hot_reload_test', str(stylesheet))
    monkeypatch.setenv(
        'NAPARI_DEV_MODULES',
        ' napari_test_styles, napari_test_styles, missing_hot_reload_test_module ',
    )

    window = make_napari_viewer().window
    widget = window._hot_reload_widget
    assert widget is not None
    assert widget.reloader.modules == (
        'napari',
        'napari_builtins',
        'napari_test_styles',
    )
    assert '#123456' in window._qt_window.styleSheet()

    stylesheet.write_text('QWidget { color: #abcdef; }', encoding='utf-8')
    qtbot.waitUntil(lambda: '#abcdef' in window._qt_window.styleSheet())


@pytest.mark.usefixtures('hot_reload_enabled')
def test_hot_reload_rebuilds_tips_in_place(
    make_napari_viewer: ViewerFactory, qtbot: QtBot
) -> None:
    from qtreload import ReloadResult

    viewer = make_napari_viewer()
    layer = viewer.add_points([[0, 0]])
    window = viewer.window
    hot_reload = window._hot_reload_widget
    assert hot_reload is not None

    _show_tips(window)
    old_widget = window.dock_widgets['Tips and Tricks']
    assert isinstance(old_widget, TipsWidget)
    dock = old_widget.parentWidget()
    assert isinstance(dock, QDockWidget)
    [live] = hot_reload.reloader.live_widgets

    with (
        qtbot.waitSignal(old_widget.destroyed),
        qtbot.waitSignal(live.evt_rebuilt),
    ):
        hot_reload.reloader.evt_batch_reloaded.emit(
            [ReloadResult(module=TipsWidget.__module__, path='', changed=True)]
        )

    new_widget = window.dock_widgets['Tips and Tricks']
    assert isinstance(new_widget, TipsWidget)
    assert new_widget is not old_widget
    assert dock.widget() is new_widget
    assert viewer.layers[0] is layer

    with qtbot.waitSignal(dock.destroyed):
        window.remove_dock_widget(dock)
    assert hot_reload.reloader.live_widgets == []
    with qtbot.waitSignal(new_widget.destroyed):
        new_widget.deleteLater()
