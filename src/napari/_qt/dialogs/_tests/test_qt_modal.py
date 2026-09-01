from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMainWindow, QPushButton, QWidget

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
        with pytest.raises(
            ValueError, match='Specifying position as a string'
        ):
            popup.move_to()

    @pytest.mark.parametrize('pos', ['top', 'bottom', 'left', 'right'])
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
        with pytest.raises(ValueError, match='position must be one of'):
            popup.move_to('dummy_text')

        with pytest.raises(TypeError, match='Wrong type of position'):
            popup.move_to({})

    @pytest.mark.parametrize('pos', [[10, 10, 10, 10], (15, 10, 10, 10)])
    def test_move_to_cords(self, pos, qtbot):
        window = QMainWindow()
        qtbot.addWidget(window)
        widget = QWidget()
        window.setCentralWidget(widget)
        popup = QtPopup(widget)
        popup.move_to(pos)

    def test_return_is_ignored_escape_closes(self, qtbot):
        popup = QtPopup(None)
        # a lone QPushButton in a QDialog is the auto-default button, which
        # QDialog.keyPressEvent would click on return
        button = QPushButton('button', popup.frame)
        clicked = MagicMock()
        button.clicked.connect(clicked)
        qtbot.addWidget(popup)
        popup.show()
        qtbot.waitUntil(popup.isVisible)

        qtbot.keyClick(popup, Qt.Key_Return)
        assert popup.isVisible()
        clicked.assert_not_called()

        qtbot.keyClick(popup, Qt.Key_Escape)
        assert not popup.isVisible()
