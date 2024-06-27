import sys

import pytest
from qtpy.QtCore import Qt

from napari._pydantic_compat import BaseModel
from napari._qt.dialogs.preferences_dialog import (
    PreferencesDialog,
    QMessageBox,
)
from napari._vendor.qt_json_builder.qt_jsonschema_form.widgets import (
    EnumSchemaWidget,
    FontSizeSchemaWidget,
    HorizontalObjectSchemaWidget,
)
from napari.settings import NapariSettings, get_settings
from napari.settings._constants import BrushSizeOnMouseModifiers, LabelDTypes


@pytest.fixture()
def pref(qtbot):
    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    settings = get_settings()
    assert settings.appearance.theme == 'dark'
    dlg._settings.appearance.theme = 'light'
    assert get_settings().appearance.theme == 'light'
    return dlg


def test_prefdialog_populated(pref):
    subfields = filter(
        lambda f: isinstance(f.type_, type) and issubclass(f.type_, BaseModel),
        NapariSettings.__fields__.values(),
    )
    assert pref._stack.count() == len(list(subfields))


def test_dask_widget(qtbot, pref):
    dask_widget = pref._stack.currentWidget().widget().widget.widgets['dask']
    def_dask_enabled = True
    settings = pref._settings

    # check custom widget definition and default value for dask cache `enabled` setting
    assert isinstance(dask_widget, HorizontalObjectSchemaWidget)
    assert settings.application.dask.enabled == def_dask_enabled
    assert dask_widget.state['enabled'] == def_dask_enabled

    # check changing dask cache `enabled` setting via widget
    new_dask_enabled = False
    dask_widget.state = {
        'enabled': new_dask_enabled,
        'cache': dask_widget.state['cache'],
    }
    assert settings.application.dask.enabled == new_dask_enabled
    assert dask_widget.state['enabled'] == new_dask_enabled
    assert dask_widget.widgets['enabled'].state == new_dask_enabled

    # check changing dask `enabled` setting via settings object (to default value)
    settings.application.dask.enabled = def_dask_enabled
    assert dask_widget.state['enabled'] == def_dask_enabled
    assert dask_widget.widgets['enabled'].state == def_dask_enabled


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


@pytest.mark.parametrize(
    ('enum_setting_name', 'enum_setting_class'),
    [
        ('new_labels_dtype', LabelDTypes),
        ('brush_size_on_mouse_move_modifiers', BrushSizeOnMouseModifiers),
    ],
)
def test_StrEnum_widgets(qtbot, pref, enum_setting_name, enum_setting_class):
    enum_widget = (
        pref._stack.currentWidget().widget().widget.widgets[enum_setting_name]
    )
    settings = pref._settings

    # check custom widget definition and widget value follows setting
    assert isinstance(enum_widget, EnumSchemaWidget)
    assert enum_widget.state == getattr(
        settings.application, enum_setting_name
    )

    # check changing setting via widget
    for idx in range(enum_widget.count()):
        item_text = enum_widget.itemText(idx)
        item_data = enum_widget.itemData(idx)
        enum_widget.setCurrentText(item_text)
        assert getattr(settings.application, enum_setting_name) == item_data
        assert enum_widget.state == item_data

    # check changing setting updates widget
    for enum_value in enum_setting_class:
        setattr(settings.application, enum_setting_name, enum_value)
        assert enum_widget.state == enum_value


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
