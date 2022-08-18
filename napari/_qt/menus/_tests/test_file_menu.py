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
