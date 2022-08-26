from unittest import mock

from npe2 import DynamicPlugin
from npe2.manifest.contributions import SampleDataURI

from napari.utils.action_manager import action_manager


def test_sample_data_triggers_reader_dialog(
    make_napari_viewer, tmp_plugin: DynamicPlugin
):
    """Sample data pops reader dialog if multiple compatible readers"""
    # make two tmp readers that take tif files
    tmp2 = tmp_plugin.spawn(register=True)

    @tmp_plugin.contribute.reader(filename_patterns=['*.tif'])
    def _(path):
        ...

    @tmp2.contribute.reader(filename_patterns=['*.tif'])
    def _(path):
        ...

    # make a sample data reader for tif file
    my_sample = SampleDataURI(
        key='tmp-sample',
        display_name='Temp Sample',
        uri='some-path/some-file.tif',
    )
    tmp_plugin.manifest.contributions.sample_data = [my_sample]

    viewer = make_napari_viewer()
    sample_action = viewer.window.file_menu.open_sample_menu.actions()[0]
    with mock.patch(
        'napari._qt.menus.file_menu.handle_gui_reading'
    ) as mock_read:
        sample_action.trigger()

    # assert that handle gui reading was called
    mock_read.assert_called_once()


def test_show_shortcuts_actions(make_napari_viewer):
    viewer = make_napari_viewer()
    assert viewer.window.file_menu._pref_dialog is None
    action_manager.trigger("napari:show_shortcuts")
    assert viewer.window.file_menu._pref_dialog is not None
    assert (
        viewer.window.file_menu._pref_dialog._list.currentItem().text()
        == "Shortcuts"
    )
    viewer.window.file_menu._pref_dialog.close()

def get_open_with_plugin_function(viewer, action_text):
    actions = viewer.window.file_menu.actions()
    open_w_plugin_menu = [action.menu() for action in actions if action.text() == 'Open with Plugin'][0]
    requested_action = [action for action in open_w_plugin_menu.actions() if action.text() == action_text][0]
    return requested_action

def test_open_file_with_plugin(make_napari_viewer):
    viewer = make_napari_viewer()
    action = get_open_with_plugin_function(viewer, 'Open File(s)...')
    with mock.patch('napari._qt.qt_viewer.QFileDialog') as mock_file, mock.patch('napari._qt.qt_viewer.QtViewer._qt_open') as mock_read:
        mock_file_instance = mock_file.return_value
        mock_file_instance.getOpenFileNames.return_value = (['my-file.tif'], '') 
        action.trigger()
    mock_read.assert_called_once_with(['my-file.tif'], stack=False, choose_plugin=True)

def test_open_file_stack_with_plugin(make_napari_viewer):
    viewer = make_napari_viewer()
    action = get_open_with_plugin_function(viewer, 'Open Files as Stack...')
    with mock.patch('napari._qt.qt_viewer.QFileDialog') as mock_file, mock.patch('napari._qt.qt_viewer.QtViewer._qt_open') as mock_read:
        mock_file_instance = mock_file.return_value
        mock_file_instance.getOpenFileNames.return_value = (['my-file.tif'], '') 
        action.trigger()
    mock_read.assert_called_once_with(['my-file.tif'], stack=True, choose_plugin=True)

def test_open_folder_with_plugin(make_napari_viewer):
    viewer = make_napari_viewer()
    action = get_open_with_plugin_function(viewer, 'Open Folder...')
    with mock.patch('napari._qt.qt_viewer.QFileDialog') as mock_file, mock.patch('napari._qt.qt_viewer.QtViewer._qt_open') as mock_read:
        mock_file_instance = mock_file.return_value
        mock_file_instance.getExistingDirectory.return_value = 'my-dir/'
        action.trigger()
    mock_read.assert_called_once_with(['my-dir/'], stack=False, choose_plugin=True)
