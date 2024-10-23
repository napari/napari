from qtpy.QtCore import Qt

from napari._qt.widgets.qt_command_palette import QCommandPalette


def test_move_command_palette(qtbot):
    qt_command_palette = QCommandPalette()
    qtbot.addWidget(qt_command_palette)
    qt_command_palette._line.setText('napari')
    assert qt_command_palette._list._selected_index == 0
    qtbot.keyClick(qt_command_palette._line, Qt.Key.Key_Down)
    assert qt_command_palette._list._selected_index == 1
    qtbot.keyClick(qt_command_palette._line, Qt.Key.Key_Up)
    assert qt_command_palette._list._selected_index == 0
    qtbot.keyClick(qt_command_palette._line, Qt.Key.Key_PageDown)
    assert qt_command_palette._list._selected_index > 3
    qtbot.keyClick(qt_command_palette._line, Qt.Key.Key_PageUp)
    assert qt_command_palette._list._selected_index == 0
