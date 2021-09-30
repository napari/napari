import pytest

from napari._qt.widgets.qt_keyboard_settings import ShortcutEditor
from napari.utils.action_manager import ActionManager


@pytest.fixture
def shortcut_editor_widget(qtbot):
    def _shortcut_editor_widget(**kwargs):
        action_manager = ActionManager()
        widget = ShortcutEditor(action_manager=action_manager, **kwargs)
        widget.show()
        qtbot.addWidget(widget)

        return widget

    return _shortcut_editor_widget


def test_shortcut_editor_defaults(
    shortcut_editor_widget,
):
    shortcut_editor_widget()
