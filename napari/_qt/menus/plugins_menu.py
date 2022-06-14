from typing import TYPE_CHECKING, Sequence

from qtpy.QtWidgets import QAction

from ...plugins import _npe2
from ...utils.translations import trans
from ..dialogs.qt_plugin_dialog import QtPluginDialog
from ._util import NapariMenu

if TYPE_CHECKING:
    from ..qt_main_window import Window


class PluginsMenu(NapariMenu):
    def __init__(self, window: 'Window'):
        self._win = window
        super().__init__(trans._('&Plugins'), window._qt_window)

        _npe2.index_npe1_adapters()
        self._build()

    def _build(self, event=None):
        self.clear()
        action = self.addAction(trans._("Install/Uninstall Plugins..."))
        action.triggered.connect(self._show_plugin_install_dialog)
        self.addSeparator()

        # Add a menu item (QAction) for each available plugin widget
        self._add_registered_widget(call_all=True)

    def _remove_unregistered_widget(self, event):
        for action in self.actions():
            if event.value in action.text():
                self.removeAction(action)
                self._win._remove_dock_widget(event=event)

    def _add_registered_widget(self, event=None, call_all=False):
        for plugin_name, widgets in _npe2.widget_iterator():
            if call_all or event.value == plugin_name:
                self._add_plugin_actions(plugin_name, widgets)

    def _add_plugin_actions(self, plugin_name: str, widgets: Sequence[str]):
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

            def _add_toggle_widget(*, key=key):
                full_name = menu_item_template.format(*key)
                if full_name in self._win._dock_widgets.keys():
                    dock_widget = self._win._dock_widgets[full_name]
                    if dock_widget.isVisible():
                        dock_widget.hide()
                    else:
                        dock_widget.show()
                    return

                self._win.add_plugin_dock_widget(*key)

            action.setCheckable(True)
            # check that this wasn't added to the menu already
            actions = [a.text() for a in menu.actions()]
            if action.text() not in actions:
                menu.addAction(action)
            action.triggered.connect(_add_toggle_widget)

    def _show_plugin_install_dialog(self):
        """Show dialog that allows users to sort the call order of plugins."""
        QtPluginDialog(self._win._qt_window).exec_()
