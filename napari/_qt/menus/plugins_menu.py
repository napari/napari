from itertools import chain
from typing import TYPE_CHECKING, Sequence

from qtpy.QtWidgets import QAction

from ...plugins import _npe2
from ...utils.translations import trans
from ..dialogs.qt_plugin_dialog import QtPluginDialog
from ..dialogs.qt_plugin_report import QtPluginErrReporter
from ._util import NapariMenu

if TYPE_CHECKING:
    from ..qt_main_window import Window


class PluginsMenu(NapariMenu):
    def __init__(self, window: 'Window'):
        self._win = window
        super().__init__(trans._('&Plugins'), window._qt_window)

        from ...plugins import plugin_manager

        _npe2.index_npe1_adapters()

        plugin_manager.discover_widgets()
        plugin_manager.events.disabled.connect(
            self._remove_unregistered_widget
        )
        plugin_manager.events.registered.connect(self._add_registered_widget)
        plugin_manager.events.unregistered.connect(
            self._remove_unregistered_widget
        )
        # Add a menu item (QAction) for each available plugin widget
        self._add_registered_widget(call_all=True)

    def _remove_unregistered_widget(self, event):

        for idx, action in enumerate(self.actions()):
            if event.value in action.text():
                self.removeAction(action)
                self._win._remove_dock_widget(event=event)

    def _add_registered_widget(self, event=None, call_all=False):
        from ...plugins import plugin_manager

        # eg ('dock', ('my_plugin', {'My widget': MyWidget}))
        for hook_type, (plugin_name, widgets) in chain(
            _npe2.widget_iterator(), plugin_manager.iter_widgets()
        ):
            if call_all or event.value == plugin_name:
                self._add_plugin_actions(hook_type, plugin_name, widgets)

    def _add_plugin_actions(
        self, hook_type: str, plugin_name: str, widgets: Sequence[str]
    ):
        from ...plugins import menu_item_template

        multiprovider = len(widgets) > 1
        if multiprovider:
            menu = NapariMenu(plugin_name, self)
            self.addMenu(menu)
        else:
            menu = self

        for wdg_name in widgets:
            key = (plugin_name, wdg_name)
            if multiprovider:
                action = QAction(wdg_name.replace("&", "&&"), parent=self)
            else:
                full_name = menu_item_template.format(*key)
                action = QAction(full_name.replace("&", "&&"), parent=self)

            def _add_toggle_widget(*, key=key, hook_type=hook_type):
                full_name = menu_item_template.format(*key)
                if full_name in self._win._dock_widgets.keys():
                    dock_widget = self._win._dock_widgets[full_name]
                    if dock_widget.isVisible():
                        dock_widget.hide()
                    else:
                        dock_widget.show()
                    return

                if hook_type == 'dock':
                    self._win.add_plugin_dock_widget(*key)
                else:
                    self._win._add_plugin_function_widget(*key)

            action.setCheckable(True)
            # check that this wasn't added to the menu already
            actions = [a.text() for a in menu.actions()]
            if action.text() not in actions:
                menu.addAction(action)
            action.triggered.connect(_add_toggle_widget)

    def _show_plugin_install_dialog(self):
        """Show dialog that allows users to sort the call order of plugins."""
        QtPluginDialog(self._win._qt_window).exec_()

    def _show_plugin_err_reporter(self):
        """Show dialog that allows users to review and report plugin errors."""
        QtPluginErrReporter(parent=self._win._qt_window).exec_()
