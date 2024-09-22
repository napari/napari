import os
from collections import defaultdict
from unittest.mock import Mock

import pytest
from qtpy.QtWidgets import QAction, QShortcut

from napari._qt.qt_event_loop import (
    _ipython_has_eventloop,
    get_app,
    get_qapp,
    run,
    set_app_id,
)


def test_qapp(qapp):
    qapp = get_qapp()
    with pytest.warns(
        FutureWarning,
        match='`QApplication` instance access through `get_app` is deprecated and will be removed in 0.6.0.\nPlease use `get_qapp` instead.\n',
    ):
        deprecated_qapp = get_app()
    assert qapp == deprecated_qapp


@pytest.mark.skipif(os.name != 'Windows', reason='Windows specific')
def test_windows_grouping_overwrite(qapp):
    import ctypes

    def get_app_id():
        mem = ctypes.POINTER(ctypes.c_wchar)()
        ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(
            ctypes.byref(mem)
        )
        res = ctypes.wstring_at(mem)
        ctypes.windll.Ole32.CoTaskMemFree(mem)
        return res

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('test_text')

    assert get_app_id() == 'test_text'
    set_app_id('custom_string')
    assert get_app_id() == 'custom_string'
    set_app_id('')
    assert get_app_id() == ''


def test_run_outside_ipython(make_napari_viewer, qapp, monkeypatch):
    """Test that we don't incorrectly give ipython the event loop."""
    assert not _ipython_has_eventloop()
    v1 = make_napari_viewer()
    assert not _ipython_has_eventloop()
    v2 = make_napari_viewer()
    assert not _ipython_has_eventloop()

    with monkeypatch.context() as m:
        mock_exec = Mock()
        m.setattr(qapp, 'exec_', mock_exec)
        run()
        mock_exec.assert_called_once()

    v1.close()
    v2.close()


def test_shortcut_collision(qtbot, make_napari_viewer):
    viewer = make_napari_viewer()
    defined_shortcuts = defaultdict(list)
    problematic_shortcuts = []
    shortcuts = viewer.window._qt_window.findChildren(QShortcut)
    for shortcut in shortcuts:
        key = shortcut.key().toString()
        if key == 'Ctrl+M':
            # menubar toggle support
            # https://github.com/napari/napari/pull/3204
            continue
        if key and key in defined_shortcuts:
            problematic_shortcuts.append(key)
        defined_shortcuts[key].append(key)

    actions = viewer.window._qt_window.findChildren(QAction)
    for action in actions:
        key = action.shortcut().toString()
        if key and key in defined_shortcuts:
            problematic_shortcuts.append(key)
        defined_shortcuts[key].append(key)
    assert not problematic_shortcuts
    # due to throttled mouse_move, a timer is started by the viewer, so we
    # need to wait for it to be done
    qtbot.wait(10)
