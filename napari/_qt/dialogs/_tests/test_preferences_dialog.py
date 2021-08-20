import pytest
from qtpy.QtCore import Qt

from napari._qt.dialogs.preferences_dialog import (
    PreferencesDialog,
    QMessageBox,
)
from napari.settings import get_settings


@pytest.fixture
def pref(qtbot):
    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    settings = get_settings()
    assert settings.appearance.theme == 'dark'
    dlg._settings.appearance.theme = 'light'
    assert get_settings().appearance.theme == 'light'
    yield dlg


def test_preferences_dialog_accept(qtbot, pref):
    with qtbot.waitSignal(pref.finished):
        pref.accept()
    assert get_settings().appearance.theme == 'light'


def test_preferences_dialog_ok(qtbot, pref):
    with qtbot.waitSignal(pref.finished):
        pref._button_ok.click()
    assert get_settings().appearance.theme == 'light'


def test_preferences_dialog_close(qtbot, pref):
    with qtbot.waitSignal(pref.finished):
        pref.close()
    assert get_settings().appearance.theme == 'light'


def test_preferences_dialog_escape(qtbot, pref):
    with qtbot.waitSignal(pref.finished):
        qtbot.keyPress(pref, Qt.Key_Escape)
    assert get_settings().appearance.theme == 'light'


def test_preferences_dialog_cancel(qtbot, pref):
    with qtbot.waitSignal(pref.finished):
        pref._button_cancel.click()
    assert get_settings().appearance.theme == 'dark'


def test_preferences_dialog_restore(qtbot, pref, monkeypatch):
    assert get_settings().appearance.theme == 'light'
    monkeypatch.setattr(
        QMessageBox, 'question', lambda *a: QMessageBox.RestoreDefaults
    )
    pref._restore_default_dialog()
    assert get_settings().appearance.theme == 'dark'
