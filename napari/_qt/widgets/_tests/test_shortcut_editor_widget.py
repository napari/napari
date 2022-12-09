from unittest.mock import patch

import pytest

from napari._qt.widgets.qt_keyboard_settings import ShortcutEditor, WarnPopup
from napari.utils.action_manager import action_manager


@pytest.fixture
def shortcut_editor_widget(qtbot):
    def _shortcut_editor_widget(**kwargs):
        widget = ShortcutEditor(**kwargs)
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
