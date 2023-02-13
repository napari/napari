import os

import numpy as np
import pytest
from npe2 import DynamicPlugin
from qtpy.QtWidgets import QLabel, QRadioButton

from napari._qt.dialogs.qt_reader_dialog import (
    QtReaderDialog,
    open_with_dialog_choices,
    prepare_remaining_readers,
)
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


def test_reader_dir_with_extension(tmpdir, reader_dialog):
    dir = tmpdir.mkdir('my_dir.zarr')
    widg = reader_dialog(pth=dir, readers={'p1': 'p1', 'p2': 'p2'})
    assert hasattr(widg, 'persist_checkbox')
    assert (
        widg.persist_checkbox.text()
        == "Remember this choice for files with a .zarr extension"
    )


def test_reader_dir(tmpdir, reader_dialog):
    dir = tmpdir.mkdir('my_dir')
    widg = reader_dialog(pth=dir, readers={'p1': 'p1', 'p2': 'p2'})
    assert (
        widg._persist_text
        == f'Remember this choice for folders labeled as {dir}{os.sep}.'
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


def test_prepare_dialog_options_no_readers():
    with pytest.raises(ReaderPluginError) as e:
        prepare_remaining_readers(
            ['my-file.fake'], 'fake-reader', RuntimeError('Reading failed')
        )
    assert 'Tried to read my-file.fake with plugin fake-reader' in str(e.value)


def test_prepare_dialog_options_multiple_plugins(builtins):
    pth = 'my-file.tif'

    readers = prepare_remaining_readers(
        [pth],
        None,
        RuntimeError(f'Multiple plugins found capable of reading {pth}'),
    )
    assert builtins.name in readers


def test_prepare_dialog_options_removes_plugin(tmp_plugin: DynamicPlugin):
    tmp2 = tmp_plugin.spawn(register=True)

    @tmp_plugin.contribute.reader(filename_patterns=['*.fake'])
    def _(path):
        ...

    @tmp2.contribute.reader(filename_patterns=['*.fake'])
    def _(path):
        ...

    readers = prepare_remaining_readers(
        ['my-file.fake'],
        tmp_plugin.name,
        RuntimeError('Reader failed'),
    )
    assert tmp2.name in readers
    assert tmp_plugin.name not in readers


def test_open_with_dialog_choices_persist(
    builtins, make_napari_viewer, tmp_path
):
    pth = tmp_path / 'my-file.npy'
    np.save(pth, np.random.random((10, 10)))

    viewer = make_napari_viewer()
    open_with_dialog_choices(
        display_name=builtins.display_name,
        persist=True,
        extension='.npy',
        readers={builtins.name: builtins.display_name},
        paths=[str(pth)],
        stack=False,
        qt_viewer=viewer.window._qt_viewer,
    )
    assert len(viewer.layers) == 1
    # make sure extension was saved with *
    assert get_settings().plugins.extension2reader['*.npy'] == builtins.name


def test_open_with_dialog_choices_raises(make_napari_viewer):
    viewer = make_napari_viewer()

    get_settings().plugins.extension2reader = {}
    with pytest.raises(ValueError):
        open_with_dialog_choices(
            display_name='Fake Plugin',
            persist=True,
            extension='.fake',
            readers={'fake-plugin': 'Fake Plugin'},
            paths=['my-file.fake'],
            stack=False,
            qt_viewer=viewer.window._qt_viewer,
        )
    # settings weren't saved because reading failed
    assert not get_settings().plugins.extension2reader
