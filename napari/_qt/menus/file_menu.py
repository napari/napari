from itertools import chain
from typing import TYPE_CHECKING

from app_model.types import Action, SubmenuItem

from napari._app_model._submenus import SUBMENUS
from napari._app_model.constants import MenuGroup, MenuId
from napari._qt._qapp_model.qactions._file import Q_FILE_ACTIONS
from napari._qt.dialogs.qt_reader_dialog import handle_gui_reading
from napari._qt.menus._util import NapariMenu
from napari.components._viewer_key_bindings import register_viewer_action
from napari.errors.reader_errors import MultipleReaderError
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari import Viewer
    from napari._qt.qt_main_window import Window


class FileMenu(NapariMenu):
    def __init__(self, window: 'Window') -> None:
        self._win = window
        super().__init__(trans._('&File'), window._qt_window)

        self._pref_dialog = None

        from napari.plugins import plugin_manager

        plugin_manager.discover_sample_data()
        plugin_manager.events.disabled.connect(self._rebuild_samples_menu)
        plugin_manager.events.registered.connect(self._rebuild_samples_menu)
        plugin_manager.events.unregistered.connect(self._rebuild_samples_menu)
        self._rebuild_samples_menu()
        self.update()

    def _rebuild_samples_menu(self):
        from napari.plugins import _npe2, menu_item_template, plugin_manager

        self.open_sample_menu.clear()

        for plugin_name, samples in chain(
            _npe2.sample_iterator(), plugin_manager._sample_data.items()
        ):
            multiprovider = len(samples) > 1
            if multiprovider:
                # use display_name for the menu item if npe2
                from npe2 import plugin_manager as pm

                try:
                    plugin_display_name = pm.get_manifest(
                        plugin_name
                    ).display_name
                except KeyError:
                    plugin_display_name = plugin_name

                sub_menu_id = f'napari/file/samples/{plugin_display_name}'
                sub_menu = (
                    MenuId.SAMPLES,
                    SubmenuItem(
                        submenu=sub_menu_id, title=trans._(plugin_display_name)
                    ),
                )
                SUBMENUS.append(sub_menu)
            else:
                sub_menu_id = MenuId.SAMPLES

            for samp_name, samp_dict in samples.items():

                def _add_sample(*_, plg=plugin_name, smp=samp_name):
                    try:
                        self._win._qt_viewer.viewer.open_sample(plg, smp)
                    except MultipleReaderError as e:
                        handle_gui_reading(
                            e.paths,
                            self._win._qt_viewer,
                            plugin_name=plg,
                            stack=False,
                        )

                display_name = samp_dict['display_name'].replace("&", "&&")
                if multiprovider:
                    title = display_name
                else:
                    title = menu_item_template.format(
                        plugin_name, display_name
                    )
                action = Action(
                    id=samp_dict['id'],
                    title=title,
                    menus=[{'id': sub_menu_id, 'group': MenuGroup.NAVIGATION}],
                    callback=_add_sample,
                )

                Q_FILE_ACTIONS.append(action)
                # menu.addAction(action)
                # action.triggered.connect(_add_sample)

    def _open_files_w_plugin(self):
        """Helper method for forcing plugin choice"""
        self._win._qt_viewer._open_files_dialog(choose_plugin=True)

    def _open_files_as_stack_w_plugin(self):
        """Helper method for forcing plugin choice"""
        self._win._qt_viewer._open_files_dialog_as_stack_dialog(
            choose_plugin=True
        )

    def _open_folder_w_plugin(self):
        """Helper method for forcing plugin choice"""
        self._win._qt_viewer._open_folder_dialog(choose_plugin=True)


@register_viewer_action(trans._("Show all key bindings"))
def show_shortcuts(viewer: 'Viewer'):
    viewer.window.file_menu._open_preferences()
    pref_list = viewer.window.file_menu._pref_dialog._list
    for i in range(pref_list.count()):
        if pref_list.item(i).text() == "Shortcuts":
            pref_list.setCurrentRow(i)
