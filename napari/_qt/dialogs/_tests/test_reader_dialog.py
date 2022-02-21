import pytest
from qtpy.QtWidgets import QLabel, QRadioButton

from napari._qt.dialogs.qt_reader_dialog import QtReaderDialog


@pytest.fixture
def reader_dialog(qtbot):
    def _reader_dialog(**kwargs):
        widget = QtReaderDialog(**kwargs)
        widget.show()
        qtbot.addWidget(widget)

        return widget

    return _reader_dialog


def test_reader_dialog_buttons(reader_dialog):
    widg = reader_dialog(
        readers={'display name': 'plugin-name', 'display 2': 'plugin2'}
    )
    assert len(widg.findChildren(QRadioButton)) == 2


def test_reader_defaults(reader_dialog, tmpdir):
    file_pth = tmpdir.join('my_file.tif')
    widg = reader_dialog(pth=file_pth, readers={'p1': 'p1', 'p2': 'p2'})

    assert widg.findChild(QLabel).text().startswith('Choose reader')
    assert widg._get_plugin_choice() == 'p1'
    assert widg.persist_checkbox.isChecked()


def test_reader_with_error_message(reader_dialog):
    widg = reader_dialog(error_message='Test Error')
    assert widg.findChild(QLabel).text().startswith('Test Error')


def test_reader_dir(tmpdir, reader_dialog):
    dir = tmpdir.mkdir('my_dir')
    widg = reader_dialog(pth=dir, readers={'p1': 'p1', 'p2': 'p2'})

    assert not hasattr(widg, 'persist_checkbox')


def test_get_plugin_choice(tmpdir, reader_dialog):
    file_pth = tmpdir.join('my_file.tif')
    widg = reader_dialog(pth=file_pth, readers={'p1': 'p1', 'p2': 'p2'})
    reader_btns = widg.reader_btn_group.buttons()

    reader_btns[1].toggle()
    assert widg._get_plugin_choice() == 'p2'

    reader_btns[0].toggle()
    assert widg._get_plugin_choice() == 'p1'


def test_get_persist_choice(tmpdir, reader_dialog):
    file_pth = tmpdir.join('my_file.tif')
    widg = reader_dialog(pth=file_pth, readers={'p1': 'p1', 'p2': 'p2'})
    assert widg._get_persist_choice()

    widg.persist_checkbox.toggle()
    assert not widg._get_persist_choice()
