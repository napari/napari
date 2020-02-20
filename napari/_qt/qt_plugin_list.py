"""Provides a QtPluginSorter that allows the user to change plugin call order.
"""
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from ..plugins import plugin_manager as napari_plugin_manager
from ..plugins.manager import permute_hook_implementations, HookOrderType
from pluggy.manager import PluginManager, _HookCaller
from typing import Optional


class QtHookImplItem(QListWidgetItem):
    """Item in a QtHookImplListWidget."""

    def __init__(self, hookimpl, parent=None):
        self.plugin_name = hookimpl.plugin_name
        if self.plugin_name == 'builtins':
            self.plugin_name = 'napari built-in'
        super().__init__(self.plugin_name, parent)
        self.hookimpl = hookimpl
        self.setBackground(Qt.black)


class QtHookImplListWidget(QListWidget):
    """A QListWidget that allows sorting of plugin hooks."""

    order_changed = Signal(list)  # emitted when the user changes the order.

    def __init__(
        self,
        hook: Optional[_HookCaller] = None,
        parent: Optional[QWidget] = None,
    ):
        """A ListWidget that shows all of the hook implementations for a spec.

        Usually instantiated by QtPluginSorter.

        Parameters
        ----------
        hook : pluggy.manager._HookCaller, optional
            A pluggy HookCaller to show implementations for. by default None
            (i.e. no hooks shown)
        parent : QWidget, optional
            Optional parent widget, by default None
        """
        super().__init__(parent)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragEnabled(True)
        self.setDragDropMode(self.InternalMove)
        self.setAcceptDrops(True)
        self.setSpacing(1)
        self.setMinimumHeight(1)
        self.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )
        self.order_changed.connect(self.permute_hook)
        self.set_hook(hook)

    def set_hook(self, hook: Optional[_HookCaller]) -> None:
        """Set the list widget to show hook implementations for ``hook``.

        Parameters
        ----------
        hook : _HookCaller, optional
            A pluggy HookCaller to show implementations for. by default None
            (i.e. no hooks shown)
        """
        self.clear()
        self.hook = hook
        if not hook:
            return
        for hookimpl in reversed(hook.get_hookimpls()):
            self.addItem(QtHookImplItem(hookimpl))

    def dropEvent(self, event):
        """Triggered when the user moves & drops one of the items in the list.

        Parameters
        ----------
        event : QEvent
            The event that triggered the dropEvent.
        """
        super().dropEvent(event)
        order = [self.item(r).hookimpl for r in range(self.count())]
        self.order_changed.emit(order)

    def permute_hook(self, order: HookOrderType):
        """Rearrage the call order of the hooks for the current hook impl.

        Parameters
        ----------
        order : list
            A list of str, hookimpls, or module_or_class, with the desired
            CALL ORDER of the hook implementations.
        """
        if not self.hook:
            return
        permute_hook_implementations(self.hook, order)


class QtPluginSorter(QDialog):
    NULL_OPTION = 'select hook... '

    def __init__(
        self,
        plugin_manager: Optional[PluginManager] = None,
        parent: Optional[QWidget] = None,
        *,
        initial_hook: Optional[str] = None,
        firstresult_only: bool = True,
    ) -> None:
        """Dialog that allows a user to change the call order of plugin hooks.

        A main QComboBox lets the user pick which hook specification they would
        like to reorder.

        Parameters
        ----------
        plugin_manager : pluggy.PluginManager, optional
            an instance of a pluggy PluginManager, by default the main
            NapariPluginManager instance
        parent : QWidget, optional
            Optional parent widget, by default None
        initial_hook : str, optional
            If provided the QComboBox at the top of the dialog will be set to
            this hook, by default None
        firstresult_only : bool, optional
            If True, only hook specifications that declare the "firstresult"
            option will be included.  (these are hooks for which only the first
            non None result is returned).  by default True (because it makes
            less sense to sort hooks where we just collect all results anyway)
            https://pluggy.readthedocs.io/en/latest/#first-result-only
        """
        plugin_manager = plugin_manager or napari_plugin_manager
        super().__init__(parent)
        self.plugin_manager = plugin_manager
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.hookComboBox = QComboBox()
        self.hookComboBox.addItem(self.NULL_OPTION)

        # populate the combo box with all of the hooks known by the plugin mngr
        hooks = []
        for name, hook_caller in vars(plugin_manager.hook).items():
            if firstresult_only:
                # if the firstresult_only option is set
                # we only want to include hook_specifications that declare the
                # "firstresult" option as True.
                if not hook_caller.spec.opts.get('firstresult', False):
                    continue
            hooks.append(name)
        self.hookComboBox.addItems(hooks)

        self.hookComboBox.activated[str].connect(self.change_hook)
        self.hookList = QtHookImplListWidget()

        title = QLabel('Plugin Sorter')
        title.setStyleSheet(
            "QLabel { background-color : rgb(38, 41, 47); "
            "color : #ddd; font-size: 18pt; }"
        )
        self.layout.addWidget(title)

        descript = QLabel(
            'select a hook to rearrange, then drag and \n'
            'drop plugins into the desired call order'
        )
        descript.setStyleSheet(
            "QLabel { background-color : rgb(38, 41, 47); "
            "color : #bbb; font-size: 12pt; }"
        )
        self.layout.addWidget(descript)

        self.layout.addWidget(self.hookComboBox)
        self.layout.addWidget(self.hookList)
        if initial_hook is not None:
            self.hookComboBox.setCurrentText(initial_hook)
            self.change_hook(initial_hook)

    def change_hook(self, hook: str) -> None:
        """Change the hook specification shown in the list.

        Parameters
        ----------
        hook : str
            Name of the new hook specification to show.
        """
        if hook == self.NULL_OPTION:
            self.hookList.set_hook(None)
        else:
            self.hookList.set_hook(getattr(self.plugin_manager.hook, hook))
