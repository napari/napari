from unittest import mock

import numpy as np
import pytest
from app_model.types import MenuItem, SubmenuItem
from npe2 import DynamicPlugin
from npe2.manifest.contributions import SampleDataURI
from qtpy.QtGui import QGuiApplication

from napari._app_model import get_app_model
from napari._app_model.constants import MenuId
from napari._qt._qapp_model._tests.utils import get_submenu_action
from napari.layers import Image
from napari.plugins._tests.test_npe2 import mock_pm  # noqa: F401
from napari.utils.action_manager import action_manager


def test_sample_data_triggers_reader_dialog(
    make_napari_viewer, tmp_plugin: DynamicPlugin
):
    """Sample data pops reader dialog if multiple compatible readers"""
    # make two tmp readers that take tif files
    tmp2 = tmp_plugin.spawn(register=True)

    @tmp_plugin.contribute.reader(filename_patterns=['*.tif'])
    def _(path): ...

    @tmp2.contribute.reader(filename_patterns=['*.tif'])
    def _(path): ...

    # make a sample data reader for tif file
    my_sample = SampleDataURI(
        key='tmp-sample',
        display_name='Temp Sample',
        uri='some-path/some-file.tif',
    )
    tmp_plugin.manifest.contributions.sample_data = [my_sample]
    app = get_app_model()
    # Configures `app`, registers actions and initializes plugins
    make_napari_viewer()
    with mock.patch(
        'napari._qt.dialogs.qt_reader_dialog.handle_gui_reading'
    ) as mock_read:
        app.commands.execute_command('tmp_plugin:tmp-sample')

    # assert that handle gui reading was called
    mock_read.assert_called_once()


def test_plugin_display_name_use_for_multiple_samples(
    make_napari_viewer,
    builtins,
):
    """Check 'display_name' used for submenu when plugin has >1 sample data."""
    app = get_app_model()
    viewer = make_napari_viewer()

    # builtins provides more than one sample,
    # so the submenu should use the `display_name` from manifest
    samples_menu = app.menus.get_menu(MenuId.FILE_SAMPLES)
    assert samples_menu[0].title == 'napari builtins'
    # Now ensure that the actions are still correct
    # trigger the action, opening the first sample: `Astronaut`
    assert 'napari:astronaut' in app.commands
    assert len(viewer.layers) == 0
    app.commands.execute_command('napari:astronaut')
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == 'astronaut'


def test_sample_menu_plugin_state_change(
    make_napari_viewer,
    tmp_plugin: DynamicPlugin,
):
    """Check samples submenu correct after plugin changes state."""

    app = get_app_model()
    pm = tmp_plugin.plugin_manager
    # Check no samples menu before plugin registration
    with pytest.raises(KeyError):
        app.menus.get_menu(MenuId.FILE_SAMPLES)

    sample1 = SampleDataURI(
        key='tmp-sample-1',
        display_name='Temp Sample One',
        uri='some-file.tif',
    )
    sample2 = SampleDataURI(
        key='tmp-sample-2',
        display_name='Temp Sample Two',
        uri='some-file.tif',
    )
    tmp_plugin.manifest.contributions.sample_data = [sample1, sample2]

    # Configures `app`, registers actions and initializes plugins
    make_napari_viewer()

    samples_menu = app.menus.get_menu(MenuId.FILE_SAMPLES)
    assert len(samples_menu) == 1
    assert isinstance(samples_menu[0], SubmenuItem)
    assert samples_menu[0].title == tmp_plugin.display_name
    samples_sub_menu = app.menus.get_menu(MenuId.FILE_SAMPLES + '/tmp_plugin')
    assert len(samples_sub_menu) == 2
    assert isinstance(samples_sub_menu[0], MenuItem)
    assert samples_sub_menu[0].command.title == 'Temp Sample One'
    assert 'tmp_plugin:tmp-sample-1' in app.commands

    # Disable plugin
    pm.disable(tmp_plugin.name)
    with pytest.raises(KeyError):
        app.menus.get_menu(MenuId.FILE_SAMPLES)
    assert 'tmp_plugin:tmp-sample-1' not in app.commands

    # Enable plugin
    pm.enable(tmp_plugin.name)
    samples_sub_menu = app.menus.get_menu(MenuId.FILE_SAMPLES + '/tmp_plugin')
    assert len(samples_sub_menu) == 2
    assert 'tmp_plugin:tmp-sample-1' in app.commands


def test_sample_menu_single_data(
    make_napari_viewer,
    tmp_plugin: DynamicPlugin,
):
    """Checks sample submenu correct when plugin has single sample data."""
    app = get_app_model()
    sample = SampleDataURI(
        key='tmp-sample-1',
        display_name='Temp Sample One',
        uri='some-file.tif',
    )
    tmp_plugin.manifest.contributions.sample_data = [sample]
    # Configures `app`, registers actions and initializes plugins
    make_napari_viewer()

    samples_menu = app.menus.get_menu(MenuId.FILE_SAMPLES)
    assert isinstance(samples_menu[0], MenuItem)
    assert len(samples_menu) == 1
    assert samples_menu[0].command.title == 'Temp Sample One (Temp Plugin)'
    assert 'tmp_plugin:tmp-sample-1' in app.commands


def test_sample_menu_sorted(
    mock_pm,  # noqa: F811
    mock_app_model,
    tmp_plugin: DynamicPlugin,
):
    from napari._app_model import get_app_model
    from napari.plugins import _initialize_plugins

    # we make sure 'plugin-b' is registered first
    tmp_plugin2 = tmp_plugin.spawn(name='plugin-b', register=True)
    tmp_plugin1 = tmp_plugin.spawn(name='plugin-a', register=True)

    @tmp_plugin1.contribute.sample_data(display_name='Sample 1')
    def sample1(): ...

    @tmp_plugin1.contribute.sample_data(display_name='Sample 2')
    def sample2(): ...

    @tmp_plugin2.contribute.sample_data(display_name='Sample 1')
    def sample2_1(): ...

    @tmp_plugin2.contribute.sample_data(display_name='Sample 2')
    def sample2_2(): ...

    _initialize_plugins()
    samples_menu = list(get_app_model().menus.get_menu('napari/file/samples'))
    submenus = [item for item in samples_menu if isinstance(item, SubmenuItem)]
    assert len(submenus) == 3
    # mock_pm registers a sample_manifest with two sample data contributions
    assert submenus[0].title == 'My Plugin'
    assert submenus[1].title == 'plugin-a'
    assert submenus[2].title == 'plugin-b'


def test_show_shortcuts_actions(make_napari_viewer):
    viewer = make_napari_viewer()
    assert viewer.window._pref_dialog is None
    action_manager.trigger('napari:show_shortcuts')
    assert viewer.window._pref_dialog is not None
    assert viewer.window._pref_dialog._list.currentItem().text() == 'Shortcuts'
    viewer.window._pref_dialog.close()


def test_image_from_clipboard(make_napari_viewer):
    make_napari_viewer()
    app = get_app_model()

    # Ensure clipboard is empty
    QGuiApplication.clipboard().clear()
    clipboard_image = QGuiApplication.clipboard().image()
    assert clipboard_image.isNull()

    # Check action command execution
    with mock.patch('napari._qt.qt_viewer.show_info') as mock_show_info:
        app.commands.execute_command(
            'napari.window.file._image_from_clipboard'
        )
    mock_show_info.assert_called_once_with('No image or link in clipboard.')


@pytest.mark.parametrize(
    ('action_id', 'dialog_method', 'dialog_return', 'filename_call', 'stack'),
    [
        (
            # Open File(s)...
            'napari.window.file.open_files_dialog',
            'getOpenFileNames',
            (['my-file.tif'], ''),
            ['my-file.tif'],
            False,
        ),
        (
            # Open Files as Stack...
            'napari.window.file.open_files_as_stack_dialog',
            'getOpenFileNames',
            (['my-file.tif'], ''),
            ['my-file.tif'],
            True,
        ),
        (
            # Open Folder...
            'napari.window.file.open_folder_dialog',
            'getExistingDirectory',
            'my-dir/',
            ['my-dir/'],
            False,
        ),
    ],
)
def test_open(
    make_napari_viewer,
    action_id,
    dialog_method,
    dialog_return,
    filename_call,
    stack,
):
    """Test base `Open ...` actions can be triggered."""
    make_napari_viewer()
    app = get_app_model()

    # Check action command execution
    with (
        mock.patch('napari._qt.qt_viewer.QFileDialog') as mock_file,
        mock.patch('napari._qt.qt_viewer.QtViewer._qt_open') as mock_read,
    ):
        mock_file_instance = mock_file.return_value
        getattr(mock_file_instance, dialog_method).return_value = dialog_return
        app.commands.execute_command(action_id)
    mock_read.assert_called_once_with(
        filename_call, stack=stack, choose_plugin=False
    )


@pytest.mark.parametrize(
    (
        'menu_str',
        'dialog_method',
        'dialog_return',
        'filename_call',
        'stack',
    ),
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
    action, _a = get_submenu_action(
        viewer.window.file_menu, 'Open with Plugin', menu_str
    )
    with (
        mock.patch('napari._qt.qt_viewer.QFileDialog') as mock_file,
        mock.patch('napari._qt.qt_viewer.QtViewer._qt_open') as mock_read,
    ):
        mock_file_instance = mock_file.return_value
        getattr(mock_file_instance, dialog_method).return_value = dialog_return
        action.trigger()
    mock_read.assert_called_once_with(
        filename_call, stack=stack, choose_plugin=True
    )


def test_preference_dialog(make_napari_viewer):
    """Test preferences action can be triggered."""
    make_napari_viewer()
    app = get_app_model()

    # Check action command execution
    with (
        mock.patch(
            'napari._qt.qt_main_window.PreferencesDialog.show'
        ) as mock_pref_dialog_show,
    ):
        app.commands.execute_command(
            'napari.window.file.show_preferences_dialog'
        )
    mock_pref_dialog_show.assert_called_once()


def test_save_layers_enablement_updated_context(make_napari_viewer, builtins):
    """Test that enablement status of save layer actions updated correctly."""
    get_app_model()
    viewer = make_napari_viewer()

    save_layers_action = viewer.window.file_menu.findAction(
        'napari.window.file.save_layers_dialog',
    )
    save_selected_layers_action = viewer.window.file_menu.findAction(
        'napari.window.file.save_layers_dialog.selected',
    )
    # Check both save actions are not enabled when no layers
    assert len(viewer.layers) == 0
    viewer.window._update_file_menu_state()
    assert not save_layers_action.isEnabled()
    assert not save_selected_layers_action.isEnabled()

    # Add selected layer and check both save actions enabled
    layer = Image(np.random.random((10, 10)))
    viewer.layers.append(layer)
    assert len(viewer.layers) == 1
    viewer.window._update_file_menu_state()
    assert save_layers_action.isEnabled()
    assert save_selected_layers_action.isEnabled()

    # Remove selection and check 'Save All Layers...' is enabled but
    # 'Save Selected Layers...' is not
    viewer.layers.selection.clear()
    viewer.window._update_file_menu_state()
    assert save_layers_action.isEnabled()
    assert not save_selected_layers_action.isEnabled()


@pytest.mark.parametrize(
    ('action_id', 'dialog_method', 'dialog_return'),
    [
        (
            # Save Selected Layers...
            'napari.window.file.save_layers_dialog.selected',
            'getSaveFileName',
            (None, None),
        ),
        (
            # Save All Layers...
            'napari.window.file.save_layers_dialog',
            'getSaveFileName',
            (None, None),
        ),
    ],
)
def test_save_layers(
    make_napari_viewer, action_id, dialog_method, dialog_return
):
    """Test save layer selected/all actions can be triggered."""
    viewer = make_napari_viewer()
    app = get_app_model()

    # Add selected layer
    layer = Image(np.random.random((10, 10)))
    viewer.layers.append(layer)
    assert len(viewer.layers) == 1
    viewer.window._update_file_menu_state()

    # Check action command execution
    with mock.patch('napari._qt.qt_viewer.QFileDialog') as mock_file:
        mock_file_instance = mock_file.return_value
        getattr(mock_file_instance, dialog_method).return_value = dialog_return
        app.commands.execute_command(action_id)
    mock_file.assert_called_once()


@pytest.mark.parametrize(
    ('action_id', 'patch_method', 'dialog_return'),
    [
        (
            # Save Screenshot with Viewer...
            'napari.window.file.save_viewer_screenshot_dialog',
            'napari._qt.dialogs.screenshot_dialog.ScreenshotDialog.exec_',
            False,
        ),
    ],
)
def test_screenshot(
    make_napari_viewer, action_id, patch_method, dialog_return
):
    """Test screenshot actions can be triggered."""
    make_napari_viewer()
    app = get_app_model()

    # Check action command execution
    with mock.patch(patch_method) as mock_screenshot:
        mock_screenshot.return_value = dialog_return
        app.commands.execute_command(action_id)
    mock_screenshot.assert_called_once()


@pytest.mark.parametrize(
    'action_id',
    [
        # Copy Screenshot with Viewer to Clipboard
        'napari.window.file.copy_viewer_screenshot',
    ],
)
def test_screenshot_to_clipboard(make_napari_viewer, qtbot, action_id):
    """Test screenshot to clipboard actions can be triggered."""
    viewer = make_napari_viewer()
    app = get_app_model()

    # Add selected layer
    layer = Image(np.random.random((10, 10)))
    viewer.layers.append(layer)
    assert len(viewer.layers) == 1
    viewer.window._update_file_menu_state()

    # Check action command execution
    # ---- Ensure clipboard is empty
    QGuiApplication.clipboard().clear()
    clipboard_image = QGuiApplication.clipboard().image()
    assert clipboard_image.isNull()
    # ---- Execute action
    with mock.patch('napari._qt.utils.add_flash_animation') as mock_flash:
        app.commands.execute_command(action_id)
    mock_flash.assert_called_once()
    # ---- Ensure clipboard has image
    clipboard_image = QGuiApplication.clipboard().image()
    assert not clipboard_image.isNull()


@pytest.mark.parametrize(
    (
        'action_id',
        'patch_method',
    ),
    [
        (
            # Restart
            'napari.window.file.restart',
            'napari._qt.qt_main_window._QtMainWindow.restart',
        ),
    ],
)
def test_restart(make_napari_viewer, action_id, patch_method):
    """Testrestart action can be triggered."""
    make_napari_viewer()
    app = get_app_model()

    # Check action command execution
    with mock.patch(patch_method) as mock_restart:
        app.commands.execute_command(action_id)
    mock_restart.assert_called_once()


@pytest.mark.parametrize(
    ('action_id', 'patch_method', 'method_params'),
    [
        (
            # Close Window
            'napari.window.file.close_dialog',
            'napari._qt.qt_main_window._QtMainWindow.close',
            (False, True),
        ),
        (
            # Exit
            'napari.window.file.quit_dialog',
            'napari._qt.qt_main_window._QtMainWindow.close',
            (True, True),
        ),
    ],
)
def test_close(make_napari_viewer, action_id, patch_method, method_params):
    """Test close/exit actions can be triggered."""
    make_napari_viewer()
    app = get_app_model()
    quit_app, confirm_need = method_params

    # Check action command execution
    with mock.patch(patch_method) as mock_close:
        app.commands.execute_command(action_id)
    mock_close.assert_called_once_with(
        quit_app=quit_app, confirm_need=confirm_need
    )
