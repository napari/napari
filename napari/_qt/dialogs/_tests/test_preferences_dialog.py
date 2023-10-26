import sys

import pytest
from qtpy.QtCore import Qt

from napari._pydantic_compat import BaseModel
from napari._qt.dialogs.preferences_dialog import (
    PreferencesDialog,
    QMessageBox,
)
from napari._vendor.qt_json_builder.qt_jsonschema_form.widgets import (
    FontSizeSchemaWidget,
    HorizontalObjectSchemaWidget,
)
from napari.settings import NapariSettings, get_settings


@pytest.fixture
def pref(qtbot):
    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    settings = get_settings()
    assert settings.appearance.theme == 'dark'
    dlg._settings.appearance.theme = 'light'
    assert get_settings().appearance.theme == 'light'
    yield dlg


def test_prefdialog_populated(pref):
    subfields = filter(
        lambda f: isinstance(f.type_, type) and issubclass(f.type_, BaseModel),
        NapariSettings.__fields__.values(),
    )
    assert pref._stack.count() == len(list(subfields))


def test_dask_widget(qtbot, pref):
    assert isinstance(
        pref._stack.currentWidget().widget().widget.widgets['dask'],
        HorizontalObjectSchemaWidget,
    )


def test_font_size_widget(qtbot, pref):
    font_size_widget = (
        pref._stack.widget(1).widget().widget.widgets['font_size']
    )
    def_font_size = 12 if sys.platform == 'darwin' else 9

    # check custom widget definition usage for the font size setting
    # and default values
    assert isinstance(font_size_widget, FontSizeSchemaWidget)
    assert get_settings().appearance.font_size == def_font_size
    assert font_size_widget.state == def_font_size

    # check setting a new font size value via widget
    new_font_size = 14
    font_size_widget.state = new_font_size
    assert get_settings().appearance.font_size == new_font_size

    # check a theme change keeps setted font size value
    assert get_settings().appearance.theme == 'light'
    get_settings().appearance.theme = 'dark'
    assert get_settings().appearance.font_size == new_font_size
    assert font_size_widget.state == new_font_size

    # check reset button works
    font_size_widget._reset_button.click()
    assert get_settings().appearance.font_size == def_font_size
    assert font_size_widget.state == def_font_size


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
