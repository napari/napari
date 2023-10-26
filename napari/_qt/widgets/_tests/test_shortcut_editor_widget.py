import sys
from unittest.mock import patch

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QApplication, QMessageBox

from napari._qt.widgets.qt_keyboard_settings import ShortcutEditor, WarnPopup
from napari.settings import get_settings
from napari.utils.action_manager import action_manager
from napari.utils.interactions import KEY_SYMBOLS


@pytest.fixture
def shortcut_editor_widget(qtbot):
    # Always reset shortcuts (settings and action manager)
    get_settings().shortcuts.reset()
    for (
        action,
        shortcuts,
    ) in get_settings().shortcuts.shortcuts.items():
        action_manager.unbind_shortcut(action)
        for shortcut in shortcuts:
            action_manager.bind_shortcut(action, shortcut)

    def _shortcut_editor_widget(**kwargs):
        widget = ShortcutEditor(**kwargs)
        widget._reset_shortcuts()
        widget.show()
        qtbot.addWidget(widget)

        return widget

    return _shortcut_editor_widget


def test_shortcut_editor_defaults(
    shortcut_editor_widget,
):
    shortcut_editor_widget()


def test_layer_actions(shortcut_editor_widget):
    widget = shortcut_editor_widget()
    assert widget.layer_combo_box.currentText() == widget.VIEWER_KEYBINDINGS
    actions1 = widget._get_layer_actions()
    assert actions1 == widget.key_bindings_strs[widget.VIEWER_KEYBINDINGS]
    widget.layer_combo_box.setCurrentText("Labels layer")
    actions2 = widget._get_layer_actions()
    assert actions2 == {**widget.key_bindings_strs["Labels layer"], **actions1}


def test_mark_conflicts(shortcut_editor_widget, qtbot):
    widget = shortcut_editor_widget()
    widget._table.item(0, widget._shortcut_col).setText("U")
    act = widget._table.item(0, widget._action_col).text()
    assert action_manager._shortcuts[act][0] == "U"
    with patch.object(WarnPopup, "exec_") as mock:
        assert not widget._mark_conflicts(action_manager._shortcuts[act][0], 1)
        assert mock.called
    assert widget._mark_conflicts("Y", 1)
    # "Y" is arbitrary chosen and on conflict with existing shortcut should be changed
    qtbot.add_widget(widget._warn_dialog)


def test_restore_defaults(shortcut_editor_widget):
    widget = shortcut_editor_widget()
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == KEY_SYMBOLS["Ctrl"]
    widget._table.item(0, widget._shortcut_col).setText("R")
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == "R"
    with patch(
        "napari._qt.widgets.qt_keyboard_settings.QMessageBox.question"
    ) as mock:
        mock.return_value = QMessageBox.RestoreDefaults
        widget._restore_button.click()
        assert mock.called
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == KEY_SYMBOLS["Ctrl"]


@pytest.mark.parametrize(
    "key, modifier, key_symbols",
    [
        (
            "U",
            Qt.KeyboardModifier.MetaModifier
            if sys.platform == "darwin"
            else Qt.KeyboardModifier.ControlModifier,
            [KEY_SYMBOLS["Ctrl"], "U"],
        ),
        (
            "Y",
            Qt.KeyboardModifier.MetaModifier
            | Qt.KeyboardModifier.ShiftModifier
            if sys.platform == "darwin"
            else Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.ShiftModifier,
            [KEY_SYMBOLS["Ctrl"], KEY_SYMBOLS["Shift"], "Y"],
        ),
    ],
)
def test_keybinding_with_modifiers(
    shortcut_editor_widget, qtbot, recwarn, key, modifier, key_symbols
):
    widget = shortcut_editor_widget()
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == KEY_SYMBOLS["Ctrl"]

    x = widget._table.columnViewportPosition(widget._shortcut_col)
    y = widget._table.rowViewportPosition(0)
    item_pos = QPoint(x, y)
    qtbot.mouseClick(
        widget._table.viewport(), Qt.MouseButton.LeftButton, pos=item_pos
    )
    qtbot.mouseDClick(
        widget._table.viewport(), Qt.MouseButton.LeftButton, pos=item_pos
    )
    qtbot.waitUntil(lambda: QApplication.focusWidget() is not None)
    qtbot.keyClicks(QApplication.focusWidget(), key, modifier=modifier)
    assert len([warn for warn in recwarn if warn.category is UserWarning]) == 0

    shortcut = widget._table.item(0, widget._shortcut_col).text()
    for key_symbol in key_symbols:
        assert key_symbol in shortcut


@pytest.mark.parametrize(
    "modifiers, key_symbols, valid",
    [
        (
            Qt.KeyboardModifier.ShiftModifier,
            [KEY_SYMBOLS["Shift"]],
            True,
        ),
        (
            Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.ShiftModifier,
            [KEY_SYMBOLS["Ctrl"]],
            False,
        ),
    ],
)
def test_keybinding_with_only_modifiers(
    shortcut_editor_widget, qtbot, recwarn, modifiers, key_symbols, valid
):
    widget = shortcut_editor_widget()
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == KEY_SYMBOLS["Ctrl"]

    x = widget._table.columnViewportPosition(widget._shortcut_col)
    y = widget._table.rowViewportPosition(0)
    item_pos = QPoint(x, y)
    qtbot.mouseClick(
        widget._table.viewport(), Qt.MouseButton.LeftButton, pos=item_pos
    )
    qtbot.mouseDClick(
        widget._table.viewport(), Qt.MouseButton.LeftButton, pos=item_pos
    )
    qtbot.waitUntil(lambda: QApplication.focusWidget() is not None)
    with patch.object(WarnPopup, "exec_") as mock:
        qtbot.keyClick(
            QApplication.focusWidget(), Qt.Key_Enter, modifier=modifiers
        )
        if valid:
            assert not mock.called
        else:
            assert mock.called
    assert len([warn for warn in recwarn if warn.category is UserWarning]) == 0

    shortcut = widget._table.item(0, widget._shortcut_col).text()
    for key_symbol in key_symbols:
        assert key_symbol in shortcut
