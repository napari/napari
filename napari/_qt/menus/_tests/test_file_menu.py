from unittest import mock

from npe2 import DynamicPlugin
from npe2.manifest.contributions import SampleDataURI


def test_sample_data_triggers_reader_dialog(
    mock_npe2_pm, tmp_reader, make_napari_viewer
):
    """Sample data pops reader dialog if multiple compatible readers"""
    # make two tmp readers that take tif files
    tmp_reader(mock_npe2_pm, 'tif-reader', filename_patterns=['*.tif'])
    tmp_reader(mock_npe2_pm, 'other-tif-reader', filename_patterns=['*.tif'])

    # make a sample data reader for tif file
    tmp_sample_plugin = DynamicPlugin('sample-plugin', mock_npe2_pm)
    my_sample = SampleDataURI(
        key='tmp-sample',
        display_name='Temp Sample',
        uri='some-path/some-file.tif',
    )
    tmp_sample_plugin.manifest.contributions.sample_data = [my_sample]
    tmp_sample_plugin.register()

    viewer = make_napari_viewer()
    sample_action = viewer.window.file_menu.open_sample_menu.actions()[0]
    with mock.patch(
        'napari._qt.menus.file_menu.handle_gui_reading'
    ) as mock_read:
        sample_action.trigger()

    # assert that handle gui reading was called
    mock_read.assert_called_once()
