import os
import subprocess
import sys
import textwrap
import warnings
from collections import defaultdict
from unittest.mock import Mock

import pytest
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QAction, QLabel, QShortcut

from napari._qt.qt_event_loop import (
    _ipython_has_eventloop,
    get_qapp,
    run,
    set_app_id,
)


@pytest.mark.skipif(os.name != 'nt', reason='Windows specific')
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
    set_app_id('')  # app id can't be an empty string
    assert get_app_id() == 'custom_string'
    set_app_id(' ')
    assert get_app_id() == ' '


@pytest.mark.parametrize('existing_app', [False, True])
def test_get_qapp_enables_opengl_context_sharing(existing_app):
    """Check startup order without reusing pytest's application or GL state."""
    code = textwrap.dedent(
        """
        import sys

        from packaging.version import parse as parse_version
        from qtpy.QtCore import Qt, qVersion
        from qtpy.QtWidgets import QApplication

        from napari._qt import qt_event_loop

        flag = Qt.ApplicationAttribute.AA_ShareOpenGLContexts
        supports_late_sharing = parse_version(qVersion()) >= parse_version(
            '6.10'
        )
        assert QApplication.instance() is None
        assert not QApplication.testAttribute(flag)
        existing_app = QApplication([]) if sys.argv[1] == 'True' else None

        class CheckedApplication(QApplication):
            def __init__(self, *args):
                assert QApplication.testAttribute(flag)
                super().__init__(*args)

        qt_event_loop.QApplication = CheckedApplication
        app = qt_event_loop.get_qapp(ipy_interactive=False)

        expected_sharing = existing_app is None or supports_late_sharing
        assert QApplication.testAttribute(flag) is expected_sharing
        assert app is QApplication.instance()
        if existing_app is not None:
            assert app is existing_app
        """
    )
    subprocess.run(
        [sys.executable, '-c', code, str(existing_app)],
        check=True,
        timeout=60,
    )


def test_run_outside_ipython(make_napari_viewer, qapp, monkeypatch):
    """Test that we don't incorrectly give ipython the event loop."""
    assert not _ipython_has_eventloop()
    v1 = make_napari_viewer()
    assert not _ipython_has_eventloop()
    v2 = make_napari_viewer()
    assert not _ipython_has_eventloop()

    with monkeypatch.context() as m:
        mock_exec = Mock()
        m.setattr(qapp, 'exec', mock_exec)
        run()
        mock_exec.assert_called_once()

    v1.close()
    v2.close()


def test_wayland_warning_on_preexisting_app(qapp, monkeypatch):
    """Warn when a pre-existing QApplication is on Wayland with Nvidia."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(qapp, 'platformName', lambda: 'wayland')
    monkeypatch.setattr(
        'napari._qt.qt_event_loop._nvidia_driver_loaded', lambda: True
    )
    with pytest.warns(UserWarning, match='Wayland startup workaround'):
        get_qapp()


@pytest.mark.parametrize(
    ('platform', 'platform_name', 'nvidia'),
    [
        ('linux', 'xcb', True),  # not on Wayland
        ('darwin', 'cocoa', True),  # not Linux
        ('linux', 'wayland', False),  # Wayland but no Nvidia driver
    ],
)
def test_no_wayland_warning(
    qapp, monkeypatch, platform, platform_name, nvidia
):
    """No warning unless on Linux+Wayland with the Nvidia driver loaded."""
    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.setattr(qapp, 'platformName', lambda: platform_name)
    monkeypatch.setattr(
        'napari._qt.qt_event_loop._nvidia_driver_loaded', lambda: nvidia
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        get_qapp()
    assert not any(
        'Wayland startup workaround' in str(w.message) for w in records
    )


def test_status_bar_requests_tabular_numerals_with_preexisting_app(
    make_napari_viewer, qapp, qtbot
):
    tag = QFont.Tag('tnum')
    app_font = QFont(qapp.font())
    host_label = QLabel('host')
    qtbot.addWidget(host_label)
    host_font = QFont(host_label.font())

    viewer = make_napari_viewer()
    window = viewer.window._qt_window
    status = window.statusBar()._status

    assert status.font().isFeatureSet(tag)
    assert qapp.font() == app_font
    assert host_label.font() == host_font

    viewer.window._update_theme()

    assert status.font().isFeatureSet(tag)


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
