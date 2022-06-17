import pytest

from napari._qt.widgets.qt_keyboard_settings import ShortcutEditor


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
