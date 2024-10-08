import pytest
from qtpy import QtWidgets as QtW
from qtpy.QtCore import Qt

from napari._qt.widgets.qt_command_palette import QCommandPalette
from napari.components.command_palette import create_napari_command_palette

PALETTE = None


@pytest.fixture
def qt_command_palette(qtbot):
    # create viewer model and command palette
    global PALETTE

    if PALETTE is None:
        PALETTE = create_napari_command_palette()
    qwidget = QtW.QWidget()
    qt_palette = PALETTE.get_widget(qwidget)
    qtbot.addWidget(qt_palette)

    yield qt_palette

    qt_palette.close()


def test_move_command_palette(qt_command_palette: QCommandPalette, qtbot):
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
