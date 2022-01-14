import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QPushButton

from napari._qt.widgets.qt_extension2reader import Extension2ReaderTable
from napari._tests.utils import restore_settings_on_exit
from napari.settings import get_settings


@pytest.fixture
def extension2reader_widget(qtbot):
    def _extension2reader_widget(**kwargs):
        widget = Extension2ReaderTable(**kwargs)
        widget.show()
        qtbot.addWidget(widget)

        return widget

    return _extension2reader_widget


def test_extension2reader_defaults(
    extension2reader_widget,
):
    with restore_settings_on_exit():
        get_settings().plugins.extension2reader = {}
        widget = extension2reader_widget()

        assert widget._table.rowCount() == 1
        assert widget._table.itemAt(0, 0).text() == 'No extensions found.'


def test_extension2reader_with_settings(
    extension2reader_widget,
):
    with restore_settings_on_exit():
        get_settings().plugins.extension2reader = {'.test': 'test-plugin'}
        widget = extension2reader_widget()

        assert widget._table.rowCount() == 1
        assert widget._table.item(0, 0).text() == '.test'
        assert (
            widget._table.cellWidget(0, 1).findChild(QLabel).text()
            == 'test-plugin'
        )


def test_extension2reader_removal(extension2reader_widget, qtbot):
    with restore_settings_on_exit():
        get_settings().plugins.extension2reader = {
            '.test': 'test-plugin',
            '.abc': 'abc-plugin',
        }
        widget = extension2reader_widget()

        assert widget._table.rowCount() == 2

        btn_to_click = widget._table.cellWidget(0, 1).findChild(QPushButton)
        qtbot.mouseClick(btn_to_click, Qt.LeftButton)

        assert get_settings().plugins.extension2reader == {
            '.abc': 'abc-plugin'
        }
        assert widget._table.rowCount() == 1
        assert widget._table.item(0, 0).text() == '.abc'
