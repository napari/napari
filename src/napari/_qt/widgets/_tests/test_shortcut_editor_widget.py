import itertools
import sys
from unittest.mock import patch

import pyautogui
import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QAbstractItemDelegate, QApplication, QMessageBox

from napari._qt.widgets.qt_keyboard_settings import ShortcutEditor, WarnPopup
from napari._tests.utils import skip_local_focus, skip_on_mac_ci
from napari.settings import get_settings
from napari.utils.action_manager import action_manager
from napari.utils.interactions import KEY_SYMBOLS
from napari.utils.key_bindings import KeyBinding

META_CONTROL_KEY = Qt.KeyboardModifier.ControlModifier
if sys.platform == 'darwin':
    META_CONTROL_KEY = Qt.KeyboardModifier.MetaModifier


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


@pytest.mark.key_bindings
def test_potentially_conflicting_actions(shortcut_editor_widget):
    widget = shortcut_editor_widget()
    assert widget.layer_combo_box.currentText() == widget.VIEWER_KEYBINDINGS
    actions1 = widget._get_potential_conflicting_actions()
    expected_actions1 = []
    for group, keybindings in widget.key_bindings_strs.items():
        expected_actions1.extend(
            zip(itertools.repeat(group), keybindings.items())
        )
    assert actions1 == expected_actions1
    widget.layer_combo_box.setCurrentText('Labels layer')
    actions2 = widget._get_potential_conflicting_actions()
    expected_actions2 = list(
        zip(
            itertools.repeat('Labels layer'),
            widget.key_bindings_strs['Labels layer'].items(),
        )
    )
    expected_actions2.extend(
        zip(
            itertools.repeat(widget.VIEWER_KEYBINDINGS),
            widget.key_bindings_strs[widget.VIEWER_KEYBINDINGS].items(),
        )
    )
    assert actions2 == expected_actions2


@pytest.mark.key_bindings
def test_mark_conflicts(shortcut_editor_widget, qtbot):
    widget = shortcut_editor_widget()
    ctrl_keybinding = KeyBinding.from_str('Ctrl')
    u_keybinding = KeyBinding.from_str('U')
    act = widget._table.item(0, widget._action_col).text()

    # Add check for initial/default keybinding (first shortcuts column) and
    # added one (second shortcuts column)
    assert action_manager._shortcuts[act][0] == ctrl_keybinding
    widget._table.item(0, widget._shortcut_col2).setText(str(u_keybinding))
    assert action_manager._shortcuts[act][1] == str(u_keybinding)

    # Check conflicts detection using `KeyBindingLike` params
    # (`KeyBinding`, `str` and `int` representations of a shortcut)
    with patch.object(WarnPopup, 'exec_') as mock:
        assert not widget._mark_conflicts(ctrl_keybinding, 1)
        assert mock.called
    with patch.object(WarnPopup, 'exec_') as mock:
        assert not widget._mark_conflicts(str(ctrl_keybinding), 1)
        assert mock.called
    with patch.object(WarnPopup, 'exec_') as mock:
        assert not widget._mark_conflicts(int(ctrl_keybinding), 1)
        assert mock.called

    with patch.object(WarnPopup, 'exec_') as mock:
        assert not widget._mark_conflicts(u_keybinding, 1)
        assert mock.called
    with patch.object(WarnPopup, 'exec_') as mock:
        assert not widget._mark_conflicts(str(u_keybinding), 1)
        assert mock.called

    # Check no conflicts are found using `KeyBindingLike` params
    # (`KeyBinding`, `str` and `int` representations of a shortcut)
    # "H" is arbitrary chosen and on conflict with existing shortcut should be changed
    h_keybinding = KeyBinding.from_str('H')
    assert widget._mark_conflicts(h_keybinding, 1)
    assert widget._mark_conflicts(str(h_keybinding), 1)
    assert widget._mark_conflicts(int(h_keybinding), 1)
    qtbot.add_widget(widget._warn_dialog)


def test_restore_defaults(shortcut_editor_widget):
    widget = shortcut_editor_widget()
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == KEY_SYMBOLS['Ctrl']
    widget._table.item(0, widget._shortcut_col).setText('H')
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == 'H'
    with patch(
        'napari._qt.widgets.qt_keyboard_settings.QMessageBox.question'
    ) as mock:
        mock.return_value = QMessageBox.RestoreDefaults
        widget._restore_button.click()
        assert mock.called
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == KEY_SYMBOLS['Ctrl']


@pytest.mark.key_bindings
@skip_local_focus
@pytest.mark.parametrize(
    ('key', 'modifier', 'key_symbols'),
    [
        (
            Qt.Key.Key_U,
            META_CONTROL_KEY,
            [KEY_SYMBOLS['Ctrl'], 'U'],
        ),
        (
            Qt.Key.Key_Y,
            META_CONTROL_KEY | Qt.KeyboardModifier.ShiftModifier,
            [KEY_SYMBOLS['Ctrl'], KEY_SYMBOLS['Shift'], 'Y'],
        ),
        (
            Qt.Key.Key_Escape,
            META_CONTROL_KEY,
            [KEY_SYMBOLS['Ctrl'], KEY_SYMBOLS['Escape']],
        ),
        (
            Qt.Key.Key_Delete,
            META_CONTROL_KEY | Qt.KeyboardModifier.ShiftModifier,
            [KEY_SYMBOLS['Ctrl'], KEY_SYMBOLS['Shift'], KEY_SYMBOLS['Delete']],
        ),
        (
            Qt.Key.Key_Backspace,
            META_CONTROL_KEY | Qt.KeyboardModifier.ShiftModifier,
            [
                KEY_SYMBOLS['Ctrl'],
                KEY_SYMBOLS['Shift'],
                KEY_SYMBOLS['Backspace'],
            ],
        ),
    ],
)
def test_keybinding_with_modifiers(
    shortcut_editor_widget, qtbot, recwarn, key, modifier, key_symbols
):
    widget = shortcut_editor_widget()
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == KEY_SYMBOLS['Ctrl']

    x = widget._table.columnViewportPosition(widget._shortcut_col)
    y = widget._table.rowViewportPosition(0)
    item_pos = QPoint(x, y)
    index = widget._table.indexAt(item_pos)
    widget._table.setCurrentIndex(index)
    widget._table.edit(index)
    qtbot.waitUntil(lambda: widget._table.focusWidget() is not None)
    editor = widget._table.focusWidget()
    qtbot.keyPress(editor, key, modifier=modifier)
    widget._table.commitData(editor)
    widget._table.closeEditor(editor, QAbstractItemDelegate.NoHint)

    assert len([warn for warn in recwarn if warn.category is UserWarning]) == 0

    shortcut = widget._table.item(0, widget._shortcut_col).text()
    for key_symbol in key_symbols:
        assert key_symbol in shortcut


@skip_local_focus
@pytest.mark.parametrize(
    ('modifiers', 'key_symbols', 'valid'),
    [
        (
            Qt.KeyboardModifier.ShiftModifier,
            [KEY_SYMBOLS['Shift']],
            True,
        ),
        (
            Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.ShiftModifier,
            [KEY_SYMBOLS['Ctrl']],
            False,
        ),
    ],
)
def test_keybinding_with_only_modifiers(
    shortcut_editor_widget, qtbot, recwarn, modifiers, key_symbols, valid
):
    widget = shortcut_editor_widget()
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == KEY_SYMBOLS['Ctrl']

    x = widget._table.columnViewportPosition(widget._shortcut_col)
    y = widget._table.rowViewportPosition(0)
    item_pos = QPoint(x, y)
    index = widget._table.indexAt(item_pos)
    widget._table.setCurrentIndex(index)
    widget._table.edit(index)
    qtbot.waitUntil(lambda: widget._table.focusWidget() is not None)
    editor = widget._table.focusWidget()

    with patch.object(WarnPopup, 'exec_') as mock:
        qtbot.keyPress(editor, Qt.Key_Enter, modifier=modifiers)
        widget._table.commitData(editor)
        widget._table.closeEditor(editor, QAbstractItemDelegate.NoHint)
        if valid:
            assert not mock.called
        else:
            assert mock.called

    assert len([warn for warn in recwarn if warn.category is UserWarning]) == 0

    shortcut = widget._table.item(0, widget._shortcut_col).text()
    for key_symbol in key_symbols:
        assert key_symbol in shortcut


@skip_local_focus
@pytest.mark.parametrize(
    'removal_trigger_key',
    [
        Qt.Key.Key_Delete,
        Qt.Key.Key_Backspace,
    ],
)
@pytest.mark.parametrize(
    'confirm_key',
    [Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Tab],
)
def test_remove_shortcut(
    shortcut_editor_widget, qtbot, removal_trigger_key, confirm_key
):
    widget = shortcut_editor_widget()
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == KEY_SYMBOLS['Ctrl']

    x = widget._table.columnViewportPosition(widget._shortcut_col)
    y = widget._table.rowViewportPosition(0)
    item_pos = QPoint(x, y)
    index = widget._table.indexAt(item_pos)
    widget._table.setCurrentIndex(index)
    widget._table.edit(index)
    qtbot.waitUntil(lambda: widget._table.focusWidget() is not None)
    editor = widget._table.focusWidget()
    qtbot.keyClick(editor, removal_trigger_key)
    qtbot.keyClick(editor, confirm_key)
    widget._table.commitData(editor)
    widget._table.closeEditor(editor, QAbstractItemDelegate.NoHint)

    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == ''


@skip_local_focus
@skip_on_mac_ci
@pytest.mark.parametrize(
    ('modifier_key', 'modifiers', 'key_symbols'),
    [
        (
            'shift',
            None,
            [KEY_SYMBOLS['Shift']],
        ),
        (
            'ctrl',
            'shift',
            [KEY_SYMBOLS['Ctrl'], KEY_SYMBOLS['Shift']],
        ),
    ],
)
def test_keybinding_editor_modifier_key_detection(
    shortcut_editor_widget,
    qtbot,
    recwarn,
    modifier_key,
    modifiers,
    key_symbols,
):
    """
    Test modifier keys detection with pyautogui to trigger keyboard events
    from the OS.

    Notes:
        * Skipped on macOS CI due to accessibility permissions not being
          settable on macOS GitHub Actions runners.
        * For this test to pass locally, you need to give the Terminal/iTerm
          application accessibility permissions:
              `System Settings > Privacy & Security > Accessibility`

        See https://github.com/asweigart/pyautogui/issues/247 and
        https://github.com/asweigart/pyautogui/issues/247#issuecomment-437668855
    """
    widget = shortcut_editor_widget()
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == KEY_SYMBOLS['Ctrl']

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

    line_edit = QApplication.focusWidget()
    with pyautogui.hold(modifier_key):
        if modifiers:
            pyautogui.keyDown(modifiers)

        def press_check():
            line_edit.selectAll()
            shortcut = line_edit.selectedText()
            all_pressed = True
            for key_symbol in key_symbols:
                all_pressed &= key_symbol in shortcut
            return all_pressed

        qtbot.waitUntil(lambda: press_check())

        if modifiers:
            pyautogui.keyUp(modifiers)

    def release_check():
        line_edit.selectAll()
        shortcut = line_edit.selectedText()
        return shortcut == ''

    qtbot.waitUntil(lambda: release_check())

    qtbot.keyClick(line_edit, Qt.Key_Escape)
    shortcut = widget._table.item(0, widget._shortcut_col).text()
    assert shortcut == KEY_SYMBOLS['Ctrl']
