from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMainWindow, QWidget

from napari._qt.dialogs.qt_modal import QtPopup


class TestQtPopup:
    def test_show_above(self, qtbot):
        popup = QtPopup(None)
        qtbot.addWidget(popup)
        popup.show_above_mouse()
        popup.close()

    def test_show_right(self, qtbot):
        popup = QtPopup(None)
        qtbot.addWidget(popup)
        popup.show_right_of_mouse()
        popup.close()

    def test_move_to_error_no_parent(self, qtbot):
        popup = QtPopup(None)
        qtbot.add_widget(popup)
        with pytest.raises(ValueError):
            popup.move_to()

    @pytest.mark.parametrize("pos", ["top", "bottom", "left", "right"])
    def test_move_to(self, pos, qtbot):
        window = QMainWindow()
        qtbot.addWidget(window)
        widget = QWidget()
        window.setCentralWidget(widget)
        popup = QtPopup(widget)
        popup.move_to(pos)

    def test_move_to_error_wrong_params(self, qtbot):
        window = QMainWindow()
        qtbot.addWidget(window)
        widget = QWidget()
        window.setCentralWidget(widget)
        popup = QtPopup(widget)
        with pytest.raises(ValueError):
            popup.move_to("dummy_text")

        with pytest.raises(ValueError):
            popup.move_to({})

    @pytest.mark.parametrize("pos", [[10, 10, 10, 10], (15, 10, 10, 10)])
    def test_move_to_cords(self, pos, qtbot):
        window = QMainWindow()
        qtbot.addWidget(window)
        widget = QWidget()
        window.setCentralWidget(widget)
        popup = QtPopup(widget)
        popup.move_to(pos)

    def test_click(self, qtbot, monkeypatch):
        popup = QtPopup(None)
        monkeypatch.setattr(popup, "close", MagicMock())
        qtbot.addWidget(popup)
        qtbot.keyClick(popup, Qt.Key_8)
        popup.close.assert_not_called()
        qtbot.keyClick(popup, Qt.Key_Return)
        popup.close.assert_called_once()
