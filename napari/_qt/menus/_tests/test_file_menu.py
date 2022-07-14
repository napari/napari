from unittest import mock

from npe2 import DynamicPlugin
from npe2.manifest.contributions import SampleDataURI

from napari._qt.menus import file_menu
from napari.settings import get_settings
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


def test_close_window_cancel(make_napari_viewer):
    v = make_napari_viewer()
    with mock.patch(
        'napari._qt.qt_main_window._QtMainWindow.close'
    ) as close_mock:
        with mock.patch(
            "napari._qt.menus.file_menu.QMessageBox.exec_",
            return_value=file_menu.QMessageBox.StandardButton.Cancel,
        ) as message_mock:
            v.window.file_menu._close_window()
            message_mock.assert_called_once()
            close_mock.assert_not_called()


def test_close_window_ok(make_napari_viewer):
    v = make_napari_viewer()
    with mock.patch(
        'napari._qt.qt_main_window._QtMainWindow.close'
    ) as close_mock:
        with mock.patch(
            "napari._qt.menus.file_menu.QMessageBox.exec_",
            return_value=file_menu.QMessageBox.StandardButton.Ok,
        ) as message_mock:
            v.window.file_menu._close_window()
            message_mock.assert_called_once()
            close_mock.assert_called_once_with(quit_app=False)


def test_close_window_no_confirm(make_napari_viewer, monkeypatch):
    v = make_napari_viewer()
    with mock.patch(
        'napari._qt.qt_main_window._QtMainWindow.close'
    ) as close_mock:
        monkeypatch.setattr(
            get_settings().application, "confirm_close_window", False
        )
        with mock.patch(
            "napari._qt.menus.file_menu.QMessageBox.exec_"
        ) as message_mock:
            v.window.file_menu._close_window()
            message_mock.assert_not_called()
            close_mock.assert_called_once_with(quit_app=False)


def test_close_app_cancel(make_napari_viewer):
    v = make_napari_viewer()
    with mock.patch(
        'napari._qt.qt_main_window._QtMainWindow.close'
    ) as close_mock:
        with mock.patch(
            "napari._qt.menus.file_menu.QMessageBox.exec_",
            return_value=file_menu.QMessageBox.StandardButton.Cancel,
        ) as message_mock:
            v.window.file_menu._close_app()
            message_mock.assert_called_once()
            close_mock.assert_not_called()


def test_close_app_ok(make_napari_viewer):
    v = make_napari_viewer()
    with mock.patch(
        'napari._qt.qt_main_window._QtMainWindow.close'
    ) as close_mock:
        with mock.patch(
            "napari._qt.menus.file_menu.QMessageBox.exec_",
            return_value=file_menu.QMessageBox.StandardButton.Ok,
        ) as message_mock:
            v.window.file_menu._close_app()
            message_mock.assert_called_once()
            close_mock.assert_called_once_with(quit_app=True)


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
