from itertools import chain
from typing import TYPE_CHECKING

from qtpy.QtCore import QSize
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QAction, QMessageBox

from napari._qt.dialogs.qt_reader_dialog import handle_gui_reading
from napari.errors.reader_errors import MultipleReaderError

from ...components._viewer_key_bindings import register_viewer_action
from ...settings import get_settings
from ...utils.history import get_save_history, update_save_history
from ...utils.misc import running_as_bundled_app
from ...utils.translations import trans
from ..dialogs.preferences_dialog import PreferencesDialog
from ..dialogs.screenshot_dialog import ScreenshotDialog
from ._util import NapariMenu, populate_menu

if TYPE_CHECKING:
    from ... import Viewer
    from ..qt_main_window import Window


class FileMenu(NapariMenu):
    def __init__(self, window: 'Window'):
        self._win = window
        super().__init__(trans._('&File'), window._qt_window)
        ]
        populate_menu(self, ACTIONS)

        self._pref_dialog = None

        from ...plugins import plugin_manager

        plugin_manager.discover_sample_data()
        plugin_manager.events.disabled.connect(self._rebuild_samples_menu)
        plugin_manager.events.registered.connect(self._rebuild_samples_menu)
        plugin_manager.events.unregistered.connect(self._rebuild_samples_menu)
        self._rebuild_samples_menu()
        self.update()


    def _rebuild_samples_menu(self):
        from ...plugins import _npe2, menu_item_template, plugin_manager

        self.open_sample_menu.clear()

        for plugin_name, samples in chain(
            _npe2.sample_iterator(), plugin_manager._sample_data.items()
        ):
            multiprovider = len(samples) > 1
            if multiprovider:
                menu = self.open_sample_menu.addMenu(plugin_name)
            else:
                menu = self.open_sample_menu

            for samp_name, samp_dict in samples.items():
                display_name = samp_dict['display_name'].replace("&", "&&")
                if multiprovider:
                    action = QAction(display_name, parent=self)
                else:
                    full_name = menu_item_template.format(
                        plugin_name, display_name
                    )
                    action = QAction(full_name, parent=self)

                def _add_sample(*args, plg=plugin_name, smp=samp_name):
                    try:
                        self._win._qt_viewer.viewer.open_sample(plg, smp)
                    except MultipleReaderError as e:
                        handle_gui_reading(
                            e.paths,
                            self._win._qt_viewer,
                            plugin_name=plugin_name,
                            stack=False,
                        )

                menu.addAction(action)
                action.triggered.connect(_add_sample)


@register_viewer_action(trans._("Show all key bindings"))
def show_shortcuts(viewer: 'Viewer'):
    viewer.window.file_menu._open_preferences()
    pref_list = viewer.window.file_menu._pref_dialog._list
    for i in range(pref_list.count()):
        if pref_list.item(i).text() == "Shortcuts":
            pref_list.setCurrentRow(i)
