import sys

import numpy.testing as npt
import pyautogui
import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QApplication

from napari._pydantic_compat import BaseModel
from napari._qt.dialogs.preferences_dialog import (
    PreferencesDialog,
    QMessageBox,
)
from napari._tests.utils import skip_local_focus, skip_on_mac_ci
from napari._vendor.qt_json_builder.qt_jsonschema_form.widgets import (
    EnumSchemaWidget,
    FontSizeSchemaWidget,
    HighlightPreviewWidget,
    HorizontalObjectSchemaWidget,
)
from napari.settings import NapariSettings, get_settings
from napari.settings._constants import BrushSizeOnMouseModifiers, LabelDTypes
from napari.utils.interactions import Shortcut
from napari.utils.key_bindings import KeyBinding


@pytest.fixture
def pref(qtbot):
    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    # check settings default values and change them for later checks
    settings = get_settings()
    # change theme setting (default `dark`)
    assert settings.appearance.theme == 'dark'
    dlg._settings.appearance.theme = 'light'
    assert get_settings().appearance.theme == 'light'
    # change highlight setting related value (default thickness `1`)
    assert get_settings().appearance.highlight.highlight_thickness == 1
    dlg._settings.appearance.highlight.highlight_thickness = 5
    assert get_settings().appearance.highlight.highlight_thickness == 5
    # change `napari:reset_scroll_progress` shortcut/keybinding (default keybinding `Ctrl`/`Control`)
    # a copy of the initial `shortcuts` dictionary needs to be done since, to trigger an
    # event update from the `ShortcutsSettings` model, the whole `shortcuts` dictionary
    # needs to be reassigned.
    assert dlg._settings.shortcuts.shortcuts[
        'napari:reset_scroll_progress'
    ] == [KeyBinding.from_str('Ctrl')]
    shortcuts = dlg._settings.shortcuts.shortcuts.copy()
    shortcuts['napari:reset_scroll_progress'] = [KeyBinding.from_str('U')]
    dlg._settings.shortcuts.shortcuts = shortcuts
    assert dlg._settings.shortcuts.shortcuts[
        'napari:reset_scroll_progress'
    ] == [KeyBinding.from_str('U')]
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

    # verify that a theme change preserves the font size value
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


def test_highlight_widget(qtbot, pref):
    highlight_widget = (
        pref._stack.widget(1).widget().widget.widgets['highlight']
    )
    settings = pref._settings

    # check custom widget definition and widget follows settings values
    assert isinstance(highlight_widget, HighlightPreviewWidget)
    assert (
        highlight_widget.state['highlight_color']
        == settings.appearance.highlight.highlight_color
    )
    assert (
        highlight_widget.state['highlight_thickness']
        == settings.appearance.highlight.highlight_thickness
    )

    # check changing setting via widget
    new_widget_values = {
        'highlight_thickness': 5,
        'highlight_color': [0.6, 0.6, 1.0, 1.0],
    }
    highlight_widget.setValue(new_widget_values)
    npt.assert_allclose(
        settings.appearance.highlight.highlight_color,
        new_widget_values['highlight_color'],
    )
    assert (
        settings.appearance.highlight.highlight_thickness
        == new_widget_values['highlight_thickness']
    )

    # check changing setting updates widget
    new_setting_values = {
        'highlight_thickness': 1,
        'highlight_color': [0.5, 0.6, 1.0, 1.0],
    }

    settings.appearance.highlight.highlight_color = new_setting_values[
        'highlight_color'
    ]
    npt.assert_allclose(
        highlight_widget.state['highlight_color'],
        new_setting_values['highlight_color'],
    )

    settings.appearance.highlight.highlight_thickness = new_setting_values[
        'highlight_thickness'
    ]
    assert (
        highlight_widget.state['highlight_thickness']
        == new_setting_values['highlight_thickness']
    )


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


@pytest.mark.key_bindings
def test_preferences_dialog_cancel(qtbot, pref):
    with qtbot.waitSignal(pref.finished):
        pref._button_cancel.click()
    assert get_settings().appearance.theme == 'dark'
    assert get_settings().shortcuts.shortcuts[
        'napari:reset_scroll_progress'
    ] == [KeyBinding.from_str('Ctrl')]


@pytest.mark.key_bindings
def test_preferences_dialog_restore(qtbot, pref, monkeypatch):
    theme_widget = pref._stack.widget(1).widget().widget.widgets['theme']
    highlight_widget = (
        pref._stack.widget(1).widget().widget.widgets['highlight']
    )
    shortcut_widget = (
        pref._stack.widget(3).widget().widget.widgets['shortcuts']
    )

    assert get_settings().appearance.theme == 'light'
    assert theme_widget.state == 'light'
    assert get_settings().appearance.highlight.highlight_thickness == 5
    assert highlight_widget.state['highlight_thickness'] == 5
    assert get_settings().shortcuts.shortcuts[
        'napari:reset_scroll_progress'
    ] == [KeyBinding.from_str('U')]
    assert KeyBinding.from_str(
        Shortcut.parse_platform(
            shortcut_widget._table.item(
                0, shortcut_widget._shortcut_col
            ).text()
        )
    ) == KeyBinding.from_str('U')

    monkeypatch.setattr(
        QMessageBox, 'question', lambda *a: QMessageBox.RestoreDefaults
    )
    pref._restore_default_dialog()

    assert get_settings().appearance.theme == 'dark'
    assert theme_widget.state == 'dark'
    assert get_settings().appearance.highlight.highlight_thickness == 1
    assert highlight_widget.state['highlight_thickness'] == 1
    assert get_settings().shortcuts.shortcuts[
        'napari:reset_scroll_progress'
    ] == [KeyBinding.from_str('Ctrl')]
    assert KeyBinding.from_str(
        Shortcut.parse_platform(
            shortcut_widget._table.item(
                0, shortcut_widget._shortcut_col
            ).text()
        )
    ) == KeyBinding.from_str('Ctrl')


@skip_local_focus
@skip_on_mac_ci
@pytest.mark.key_bindings
@pytest.mark.parametrize(
    'confirm_key',
    ['enter', 'return', 'tab'],
)
def test_preferences_dialog_not_dismissed_by_keybind_confirm(
    qtbot, pref, confirm_key
):
    """This test ensures that when confirming a keybinding change, the dialog is not dismissed.

    Notes:
        * Skipped on macOS CI due to accessibility permissions not being
          settable on macOS GitHub Actions runners.
        * For this test to pass locally, you need to give the Terminal/iTerm/VSCode
          application accessibility permissions:
              `System Settings > Privacy & Security > Accessibility`

        See https://github.com/asweigart/pyautogui/issues/247 and
        https://github.com/asweigart/pyautogui/issues/247#issuecomment-437668855
    """
    shortcut_widget = (
        pref._stack.widget(3).widget().widget.widgets['shortcuts']
    )
    pref._stack.setCurrentIndex(3)
    # ensure the dialog is showing
    pref.show()
    qtbot.waitExposed(pref)
    assert pref.isVisible()

    shortcut = shortcut_widget._table.item(
        0, shortcut_widget._shortcut_col
    ).text()
    assert shortcut == 'U'

    x = shortcut_widget._table.columnViewportPosition(
        shortcut_widget._shortcut_col
    )
    y = shortcut_widget._table.rowViewportPosition(0)

    item_pos = QPoint(x, y)
    qtbot.mouseClick(
        shortcut_widget._table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=item_pos,
    )
    qtbot.mouseDClick(
        shortcut_widget._table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=item_pos,
    )
    qtbot.waitUntil(lambda: QApplication.focusWidget() is not None)
    pyautogui.press('delete')
    qtbot.wait(100)
    pyautogui.press(confirm_key)
    qtbot.wait(100)

    # ensure the dialog is still open
    assert pref.isVisible()

    # verify that the keybind is changed
    shortcut = shortcut_widget._table.item(
        0, shortcut_widget._shortcut_col
    ).text()
    assert shortcut == ''
