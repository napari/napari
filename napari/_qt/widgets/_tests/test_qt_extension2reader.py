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
        assert (
            widget._table.itemAt(0, 0).text()
            == 'No filename preferences found.'
        )


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

        # remove remaining extension
        btn_to_click = widget._table.cellWidget(0, 1).findChild(QPushButton)
        qtbot.mouseClick(btn_to_click, Qt.LeftButton)
        assert get_settings().plugins.extension2reader == {}
        assert widget._table.rowCount() == 1
        assert (
            "No filename preferences found" in widget._table.item(0, 0).text()
        )


def test_all_readers_in_dropdown(
    extension2reader_widget, qtbot, tmp_reader, mock_npe2_pm
):
    tmp_reader(mock_npe2_pm, 'npe2-name', filename_patterns=['*'])
    tmp_reader(mock_npe2_pm, 'other-reader', filename_patterns=['*.tif'])

    npe2_readers = {
        'npe2-name': 'npe2 Display',
        'other-reader': 'Other Reader',
    }
    npe1_readers = {'builtins': 'builtins'}

    widget = extension2reader_widget(
        npe2_readers=npe2_readers, npe1_readers=npe1_readers
    )
    all_reader_display_names = list(
        dict(npe2_readers, **npe1_readers).values()
    )
    all_dropdown_items = [
        widget._new_reader_dropdown.itemText(i)
        for i in range(widget._new_reader_dropdown.count())
    ]
    assert sorted(all_reader_display_names) == sorted(all_dropdown_items)


def test_directory_readers_not_in_dropdown(
    extension2reader_widget, qtbot, tmp_reader, mock_npe2_pm
):
    tmp_reader(
        mock_npe2_pm,
        'dir-reader',
        filename_patterns=[],
        accepts_directories=True,
    )

    widget = extension2reader_widget(
        npe2_readers={'dir-reader': 'Directory Reader'}, npe1_readers={}
    )
    all_dropdown_items = [
        widget._new_reader_dropdown.itemText(i)
        for i in range(widget._new_reader_dropdown.count())
    ]
    assert 'Directory Reader' not in all_dropdown_items


def test_filtering_readers(
    extension2reader_widget, qtbot, tmp_reader, mock_npe2_pm
):
    tmp_reader(mock_npe2_pm, 'npy-reader', filename_patterns=['*.npy'])
    tmp_reader(mock_npe2_pm, 'tif-reader', filename_patterns=['*.tif'])

    widget = extension2reader_widget(npe1_readers={'builtins': 'builtins'})

    assert widget._new_reader_dropdown.count() == 3
    widget._filter_compatible_readers('*.npy')
    assert widget._new_reader_dropdown.count() == 2
    all_dropdown_items = [
        widget._new_reader_dropdown.itemText(i)
        for i in range(widget._new_reader_dropdown.count())
    ]
    assert sorted(['npy-reader', 'builtins']) == all_dropdown_items


def test_filtering_readers_complex_pattern(
    extension2reader_widget, qtbot, tmp_reader, mock_npe2_pm
):
    tmp_reader(mock_npe2_pm, 'npy-reader', filename_patterns=['*.npy'])
    tmp_reader(
        mock_npe2_pm,
        'tif-reader',
        filename_patterns=['my-specific-folder/*.tif'],
    )

    widget = extension2reader_widget(npe1_readers={})

    assert widget._new_reader_dropdown.count() == 2
    widget._filter_compatible_readers('my-specific-folder/my-file.tif')
    assert widget._new_reader_dropdown.count() == 1
    all_dropdown_items = [
        widget._new_reader_dropdown.itemText(i)
        for i in range(widget._new_reader_dropdown.count())
    ]
    assert sorted(['tif-reader']) == all_dropdown_items


def test_adding_new_preference(
    extension2reader_widget, qtbot, tmp_reader, mock_npe2_pm
):
    tmp_reader(mock_npe2_pm, 'npy-reader', filename_patterns=['*.npy'])
    tif_reader = tmp_reader(
        mock_npe2_pm, 'tif-reader', filename_patterns=['*.tif']
    )
    tif_reader.manifest.display_name = "TIF Reader"

    widget = extension2reader_widget(npe1_readers={})
    widget._fn_pattern_edit.setText('*.tif')
    # will be filtered and tif-reader will be only item
    widget._new_reader_dropdown.setCurrentIndex(0)

    with restore_settings_on_exit():
        get_settings().plugins.extension2reader = {}
        widget._save_new_preference(True)
        settings = get_settings().plugins.extension2reader
        assert '*.tif' in settings
        assert settings['*.tif'] == 'tif-reader'
        assert (
            widget._table.item(widget._table.rowCount() - 1, 0).text()
            == '*.tif'
        )
        plugin_label = widget._table.cellWidget(
            widget._table.rowCount() - 1, 1
        ).findChild(QLabel)
        assert plugin_label.text() == 'TIF Reader'


def test_adding_new_preference_no_asterisk(
    extension2reader_widget, qtbot, tmp_reader, mock_npe2_pm
):
    tmp_reader(mock_npe2_pm, 'npy-reader', filename_patterns=['*.npy'])
    tif_reader = tmp_reader(
        mock_npe2_pm, 'tif-reader', filename_patterns=['*.tif']
    )
    tif_reader.manifest.display_name = "TIF Reader"

    widget = extension2reader_widget(npe1_readers={})
    widget._fn_pattern_edit.setText('.tif')
    # will be filtered and tif-reader will be only item
    widget._new_reader_dropdown.setCurrentIndex(0)

    with restore_settings_on_exit():
        get_settings().plugins.extension2reader = {}
        widget._save_new_preference(True)
        settings = get_settings().plugins.extension2reader
        assert '*.tif' in settings
        assert settings['*.tif'] == 'tif-reader'


def test_editing_preference(
    extension2reader_widget, qtbot, tmp_reader, mock_npe2_pm
):
    tmp_reader(mock_npe2_pm, 'tif-reader', filename_patterns=['*.tif'])
    tmp_reader(mock_npe2_pm, 'other-tif-reader', filename_patterns=['*.tif'])

    with restore_settings_on_exit():
        get_settings().plugins.extension2reader = {'*.tif': 'tif-reader'}

        widget = extension2reader_widget(npe1_readers={})
        widget._fn_pattern_edit.setText('*.tif')
        # will be filtered and other-tif-reader will be first item
        widget._new_reader_dropdown.setCurrentIndex(0)
        original_row_count = widget._table.rowCount()
        widget._save_new_preference(True)
        settings = get_settings().plugins.extension2reader
        assert '*.tif' in settings
        assert settings['*.tif'] == 'other-tif-reader'
        assert widget._table.rowCount() == original_row_count
        plugin_label = widget._table.cellWidget(
            original_row_count - 1, 1
        ).findChild(QLabel)
        assert plugin_label.text() == 'other-tif-reader'
