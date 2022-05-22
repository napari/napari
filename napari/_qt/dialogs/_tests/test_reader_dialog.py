import numpy as np
import pytest
from qtpy.QtWidgets import QLabel, QRadioButton

from napari._qt.dialogs.qt_reader_dialog import (
    QtReaderDialog,
    open_with_dialog_choices,
    prepare_remaining_readers,
)
from napari._tests.utils import restore_settings_on_exit
from napari.errors.reader_errors import ReaderPluginError
from napari.settings import get_settings


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


def test_reader_dir_with_extension(tmpdir, reader_dialog):
    dir = tmpdir.mkdir('my_dir.zarr')
    widg = reader_dialog(pth=dir, readers={'p1': 'p1', 'p2': 'p2'})
    assert hasattr(widg, 'persist_checkbox')
    assert (
        widg.persist_checkbox.text()
        == "Remember this choice for files with a .zarr extension"
    )


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


def test_prepare_dialog_options_no_readers(mock_npe2_pm):
    pth = 'my-file.fake'

    with pytest.raises(ReaderPluginError) as e:
        prepare_remaining_readers(
            [pth], 'fake-reader', RuntimeError('Reading failed')
        )
    assert 'Tried to read my-file.fake with plugin fake-reader' in str(e.value)


def test_prepare_dialog_options_multiple_plugins(mock_npe2_pm):
    pth = 'my-file.tif'

    readers = prepare_remaining_readers(
        [pth],
        None,
        RuntimeError(f'Multiple plugins found capable of reading {pth}'),
    )
    assert 'builtins' in readers


def test_prepare_dialog_options_removes_plugin(mock_npe2_pm, tmp_reader):
    pth = 'my-file.fake'

    tmp_reader(mock_npe2_pm, 'fake-reader')
    tmp_reader(mock_npe2_pm, 'other-fake-reader')
    readers = prepare_remaining_readers(
        [pth], 'fake-reader', RuntimeError('Reader failed')
    )
    assert 'other-fake-reader' in readers
    assert 'fake-reader' not in readers


def test_open_with_dialog_choices_persist(make_napari_viewer, tmp_path):
    pth = tmp_path / 'my-file.npy'
    np.save(pth, np.random.random((10, 10)))
    display_name = 'builtins'
    persist = True
    extension = '.npy'
    readers = {'builtins': 'builtins'}
    paths = [str(pth)]
    stack = False

    with restore_settings_on_exit():
        viewer = make_napari_viewer()
        open_with_dialog_choices(
            display_name,
            persist,
            extension,
            readers,
            paths,
            stack,
            viewer.window._qt_viewer,
        )
        assert len(viewer.layers) == 1
        # make sure extension was saved with *
        assert get_settings().plugins.extension2reader['*.npy'] == 'builtins'


def test_open_with_dialog_choices_raises(make_napari_viewer):
    pth = 'my-file.fake'
    display_name = 'Fake Plugin'
    persist = True
    extension = '.fake'
    readers = {'fake-plugin': 'Fake Plugin'}
    paths = [str(pth)]
    stack = False

    with restore_settings_on_exit():
        viewer = make_napari_viewer()
        get_settings().plugins.extension2reader = {}
        with pytest.raises(ValueError):
            open_with_dialog_choices(
                display_name,
                persist,
                extension,
                readers,
                paths,
                stack,
                viewer.window._qt_viewer,
            )
        # settings weren't saved because reading failed
        get_settings().plugins.extension2reader == {}
