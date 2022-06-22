import pytest
from npe2 import DynamicPlugin
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QPushButton

from napari._qt.widgets.qt_extension2reader import Extension2ReaderTable
from napari.settings import get_settings


@pytest.fixture
def extension2reader_widget(qtbot):
    def _extension2reader_widget(**kwargs):
        widget = Extension2ReaderTable(**kwargs)
        widget.show()
        qtbot.addWidget(widget)

        return widget

    return _extension2reader_widget


@pytest.fixture
def tif_reader(tmp_plugin: DynamicPlugin):
    tmp2 = tmp_plugin.spawn(name='tif_reader', register=True)

    @tmp2.contribute.reader(filename_patterns=['*.tif'])
    def _(path):
        ...

    return tmp2


@pytest.fixture
def npy_reader(tmp_plugin: DynamicPlugin):
    tmp2 = tmp_plugin.spawn(name='npy_reader', register=True)

    @tmp2.contribute.reader(filename_patterns=['*.npy'])
    def _(path):
        ...

    return tmp2


def test_extension2reader_defaults(
    extension2reader_widget,
):
    get_settings().plugins.extension2reader = {}
    widget = extension2reader_widget()

    assert widget._table.rowCount() == 1
    assert (
        widget._table.itemAt(0, 0).text() == 'No filename preferences found.'
    )


def test_extension2reader_with_settings(
    extension2reader_widget,
):
    get_settings().plugins.extension2reader = {'.test': 'test-plugin'}
    widget = extension2reader_widget()

    assert widget._table.rowCount() == 1
    assert widget._table.item(0, 0).text() == '.test'
    assert (
        widget._table.cellWidget(0, 1).findChild(QLabel).text()
        == 'test-plugin'
    )


def test_extension2reader_removal(extension2reader_widget, qtbot):
    get_settings().plugins.extension2reader = {
        '.test': 'test-plugin',
        '.abc': 'abc-plugin',
    }
    widget = extension2reader_widget()

    assert widget._table.rowCount() == 2

    btn_to_click = widget._table.cellWidget(0, 1).findChild(QPushButton)
    qtbot.mouseClick(btn_to_click, Qt.LeftButton)

    assert get_settings().plugins.extension2reader == {'.abc': 'abc-plugin'}
    assert widget._table.rowCount() == 1
    assert widget._table.item(0, 0).text() == '.abc'

    # remove remaining extension
    btn_to_click = widget._table.cellWidget(0, 1).findChild(QPushButton)
    qtbot.mouseClick(btn_to_click, Qt.LeftButton)
    assert not get_settings().plugins.extension2reader
    assert widget._table.rowCount() == 1
    assert "No filename preferences found" in widget._table.item(0, 0).text()


def test_all_readers_in_dropdown(
    extension2reader_widget, tmp_plugin, tif_reader
):
    @tmp_plugin.contribute.reader(filename_patterns=['*'])
    def _(path):
        ...

    npe2_readers = {
        tmp_plugin.name: tmp_plugin.display_name,
        tif_reader.name: tif_reader.display_name,
    }

    widget = extension2reader_widget(npe2_readers=npe2_readers)
    all_reader_display_names = list(dict(npe2_readers).values())
    all_dropdown_items = [
        widget._new_reader_dropdown.itemText(i)
        for i in range(widget._new_reader_dropdown.count())
    ]
    assert sorted(all_reader_display_names) == sorted(all_dropdown_items)


def test_directory_readers_not_in_dropdown(
    extension2reader_widget, tmp_plugin
):
    @tmp_plugin.contribute.reader(
        filename_patterns=[], accepts_directories=True
    )
    def f(path):
        ...

    widget = extension2reader_widget(
        npe2_readers={tmp_plugin.name: tmp_plugin.display_name},
        npe1_readers={},
    )
    all_dropdown_items = [
        widget._new_reader_dropdown.itemText(i)
        for i in range(widget._new_reader_dropdown.count())
    ]
    assert tmp_plugin.display_name not in all_dropdown_items


def test_filtering_readers(
    extension2reader_widget, builtins, tif_reader, npy_reader
):
    widget = extension2reader_widget(
        npe1_readers={builtins.display_name: builtins.display_name}
    )

    assert widget._new_reader_dropdown.count() == 3
    widget._filter_compatible_readers('*.npy')
    assert widget._new_reader_dropdown.count() == 2
    all_dropdown_items = [
        widget._new_reader_dropdown.itemText(i)
        for i in range(widget._new_reader_dropdown.count())
    ]
    assert (
        sorted([npy_reader.display_name, builtins.display_name])
        == all_dropdown_items
    )


def test_filtering_readers_complex_pattern(
    extension2reader_widget, npy_reader, tif_reader
):
    @tif_reader.contribute.reader(
        filename_patterns=['my-specific-folder/*.tif']
    )
    def f(path):
        ...

    widget = extension2reader_widget(npe1_readers={})

    assert widget._new_reader_dropdown.count() == 2
    widget._filter_compatible_readers('my-specific-folder/my-file.tif')
    assert widget._new_reader_dropdown.count() == 1
    all_dropdown_items = [
        widget._new_reader_dropdown.itemText(i)
        for i in range(widget._new_reader_dropdown.count())
    ]
    assert sorted([tif_reader.name]) == all_dropdown_items


def test_adding_new_preference(
    extension2reader_widget, tif_reader, npy_reader
):

    widget = extension2reader_widget(npe1_readers={})
    widget._fn_pattern_edit.setText('*.tif')
    # will be filtered and tif-reader will be only item
    widget._new_reader_dropdown.setCurrentIndex(0)

    get_settings().plugins.extension2reader = {}
    widget._save_new_preference(True)
    settings = get_settings().plugins.extension2reader
    assert '*.tif' in settings
    assert settings['*.tif'] == tif_reader.name
    assert (
        widget._table.item(widget._table.rowCount() - 1, 0).text() == '*.tif'
    )
    plugin_label = widget._table.cellWidget(
        widget._table.rowCount() - 1, 1
    ).findChild(QLabel)
    assert plugin_label.text() == tif_reader.display_name


def test_adding_new_preference_no_asterisk(
    extension2reader_widget, tif_reader, npy_reader
):

    widget = extension2reader_widget(npe1_readers={})
    widget._fn_pattern_edit.setText('.tif')
    # will be filtered and tif-reader will be only item
    widget._new_reader_dropdown.setCurrentIndex(0)

    get_settings().plugins.extension2reader = {}
    widget._save_new_preference(True)
    settings = get_settings().plugins.extension2reader
    assert '*.tif' in settings
    assert settings['*.tif'] == tif_reader.name


def test_editing_preference(extension2reader_widget, tif_reader):
    tiff2 = tif_reader.spawn(register=True)

    @tiff2.contribute.reader(filename_patterns=["*.tif"])
    def ff(path):
        ...

    get_settings().plugins.extension2reader = {'*.tif': tif_reader.name}

    widget = extension2reader_widget()
    widget._fn_pattern_edit.setText('*.tif')
    # set to tiff2
    widget._new_reader_dropdown.setCurrentText(tiff2.display_name)
    original_row_count = widget._table.rowCount()
    widget._save_new_preference(True)
    settings = get_settings().plugins.extension2reader
    assert '*.tif' in settings
    assert settings['*.tif'] == tiff2.name
    assert widget._table.rowCount() == original_row_count
    plugin_label = widget._table.cellWidget(
        original_row_count - 1, 1
    ).findChild(QLabel)
    assert plugin_label.text() == tiff2.name
