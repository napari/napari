from unittest import mock

import pytest
import qtpy
from npe2 import DynamicPlugin
from npe2.manifest.contributions import SampleDataURI
from qtpy.QtWidgets import QMenu

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


def test_plugin_display_name_use_for_multiple_samples(
    make_napari_viewer, builtins
):
    """For plugin with more than two sample datasets, should use plugin_display for building the menu"""
    viewer = make_napari_viewer()
    # builtins provides more than one sample, so the submenu should use the `display_name` from manifest
    plugin_action_menu = viewer.window.file_menu.open_sample_menu.actions()[
        0
    ].menu()
    assert plugin_action_menu.title() == 'napari builtins'
    # Now ensure that the actions are still correct
    # trigger the action, opening the first sample: `Astronaut`
    assert len(viewer.layers) == 0
    plugin_action_menu.actions()[0].trigger()
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == 'astronaut'


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


def get_open_with_plugin_action(viewer, action_text):
    def _get_menu(act):
        # this function may be removed when PyQt6 will release next version
        # (after 6.3.1 - if we do not want to support this test on older PyQt6)
        # https://www.riverbankcomputing.com/pipermail/pyqt/2022-July/044817.html
        # because both PyQt6 and PySide6 will have working manu method of action
        return (
            QMenu.menuInAction(act)
            if getattr(qtpy, 'PYQT6', False)
            else act.menu()
        )

    actions = viewer.window.file_menu.actions()
    for action1 in actions:
        if action1.text() == 'Open with Plugin':
            for action2 in _get_menu(action1).actions():
                if action2.text() == action_text:
                    return action2, action1
    raise ValueError(
        f'Could not find action "{action_text}"'
    )  # pragma: no cover


@pytest.mark.parametrize(
    "menu_str,dialog_method,dialog_return,filename_call,stack",
    [
        (
            'Open File(s)...',
            'getOpenFileNames',
            (['my-file.tif'], ''),
            ['my-file.tif'],
            False,
        ),
        (
            'Open Files as Stack...',
            'getOpenFileNames',
            (['my-file.tif'], ''),
            ['my-file.tif'],
            True,
        ),
        (
            'Open Folder...',
            'getExistingDirectory',
            'my-dir/',
            ['my-dir/'],
            False,
        ),
    ],
)
def test_open_with_plugin(
    make_napari_viewer,
    menu_str,
    dialog_method,
    dialog_return,
    filename_call,
    stack,
):
    viewer = make_napari_viewer()
    action, _a = get_open_with_plugin_action(viewer, menu_str)
    with mock.patch(
        'napari._qt.qt_viewer.QFileDialog'
    ) as mock_file, mock.patch(
        'napari._qt.qt_viewer.QtViewer._qt_open'
    ) as mock_read:
        mock_file_instance = mock_file.return_value
        getattr(mock_file_instance, dialog_method).return_value = dialog_return
        action.trigger()
    mock_read.assert_called_once_with(
        filename_call, stack=stack, choose_plugin=True
    )
