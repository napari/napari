from typing import TYPE_CHECKING

from qtpy.QtWidgets import QAction, QMenu

from ...utils.translations import trans
from ..dialogs.qt_plugin_dialog import QtPluginDialog
from ..dialogs.qt_plugin_report import QtPluginErrReporter

if TYPE_CHECKING:
    from ..qt_main_window import Window


class PluginsMenu(QMenu):
    def __init__(self, window: 'Window'):
        self._win = window
        super().__init__(trans._('&Plugins'), window._qt_window)

        from ...plugins import plugin_manager

        plugin_manager.discover_widgets()
        plugin_manager.events.disabled.connect(self._rebuild)
        plugin_manager.events.registered.connect(self._rebuild)
        plugin_manager.events.unregistered.connect(self._rebuild)
        self._rebuild()

    def _rebuild(self, event=None):
        from ...plugins import menu_item_template, plugin_manager

        self.clear()
        action = self.addAction(trans._("Install/Uninstall Plugins..."))
        action.triggered.connect(self._show_plugin_install_dialog)
        action = self.addAction(trans._("Plugin Errors..."))
        action.setStatusTip(
            trans._(
                'Review stack traces for plugin exceptions and notify developers'
            )
        )
        action.triggered.connect(self._show_plugin_err_reporter)
        self.addSeparator()

        # Add a menu item (QAction) for each available plugin widget
        for hook_type, (plugin_name, widgets) in plugin_manager.iter_widgets():
            multiprovider = len(widgets) > 1
            if multiprovider:
                menu = QMenu(plugin_name, self)
                self.addMenu(menu)
            else:
                menu = self

            for wdg_name in widgets:
                key = (plugin_name, wdg_name)
                if multiprovider:
                    action = QAction(wdg_name, parent=self)
                else:
                    full_name = menu_item_template.format(*key)
                    action = QAction(full_name, parent=self)

                def _add_widget(*args, key=key, hook_type=hook_type):
                    if hook_type == 'dock':
                        self._win.add_plugin_dock_widget(*key)
                    else:
                        self._win._add_plugin_function_widget(*key)

                menu.addAction(action)
                action.triggered.connect(_add_widget)

    def _show_plugin_install_dialog(self):
        """Show dialog that allows users to sort the call order of plugins."""
        QtPluginDialog(self._win._qt_window).exec_()

    def _show_plugin_err_reporter(self):
        """Show dialog that allows users to review and report plugin errors."""
        QtPluginErrReporter(parent=self._win._qt_window).exec_()
