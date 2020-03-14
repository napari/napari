"""Provides a QtPluginSorter that allows the user to change plugin call order.
"""
from typing import Optional

from pluggy.manager import PluginManager, _HookCaller
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QDrag, QPixmap, QPainter, QCursor
from qtpy.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..plugins import plugin_manager as napari_plugin_manager
from ..plugins.utils import HookOrderType, permute_hook_implementations


class ImplementationListItem(QFrame):
    def __init__(self, item, position=0, parent=None):
        super().__init__(parent)
        self.item = item
        self.opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.positionLabel = QLabel()
        self.update_position_label()

        self.label = QLabel(item.hookimpl.plugin_name)
        self.activeCheckBox = QCheckBox(self)
        self.activeCheckBox.stateChanged.connect(self._set_enabled)
        self.activeCheckBox.setChecked(getattr(item.hookimpl, 'enabled', True))
        layout.addWidget(self.positionLabel)
        layout.addWidget(self.activeCheckBox)
        layout.addWidget(self.label)
        layout.setStretch(2, 1)
        layout.setContentsMargins(0, 0, 0, 0)

    def _set_enabled(self, state):
        self.opacity.setOpacity(1 if state else 0.5)
        # "hookimpl.enabled" is NOT a pluggy attribute... we are adding that to
        # allow skipping of hook_implementations.
        # see plugins.io.read_data_with_plugins() for an example
        setattr(self.item.hookimpl, 'enabled', bool(state))

    def update_position_label(self):
        position = self.item.listWidget().indexFromItem(self.item).row() + 1
        self.positionLabel.setText(str(position))


class QtHookImplListWidget(QListWidget):
    """A QListWidget that allows sorting of plugin hooks."""

    order_changed = Signal(list)  # emitted when the user changes the order.

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        hook: Optional[_HookCaller] = None,
    ):
        """A ListWidget that shows all of the hook implementations for a spec.

        Usually instantiated by QtPluginSorter.

        Parameters
        ----------
        parent : QWidget, optional
            Optional parent widget, by default None
        hook : pluggy.manager._HookCaller, optional
            A pluggy HookCaller to show implementations for. by default None
            (i.e. no hooks shown)
        """
        super().__init__(parent)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragEnabled(True)
        self.setDragDropMode(self.InternalMove)
        self.setSelectionMode(self.SingleSelection)
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
            self.add_hook_implementation_to_list(hookimpl)

    def add_hook_implementation_to_list(self, hookimpl):
        # don't want users to be able to resort builtin plugins.
        # this may change in the future, and might require hook-specific rules
        if hookimpl.plugin_name == 'builtins':
            return
        item = QListWidgetItem(parent=self)
        item.hookimpl = hookimpl
        self.addItem(item)
        widg = ImplementationListItem(item)
        item.setSizeHint(widg.sizeHint())
        self.order_changed.connect(widg.update_position_label)
        self.setItemWidget(item, widg)

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

    def startDrag(self, supportedActions):
        drag = drag_with_pixmap(self)
        drag.exec_(supportedActions, Qt.MoveAction)

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
        self.hookList = QtHookImplListWidget(parent=self)

        title = QLabel('Plugin Sorter')
        title.setObjectName("h2")
        self.layout.addWidget(title)

        instructions = QLabel(
            'Select a hook to rearrange, then drag and '
            'drop plugins into the desired call order. '
            '\nDisable plugins by unchecking their checkbox.'
        )
        instructions.setWordWrap(True)
        self.layout.addWidget(instructions)

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


def drag_with_pixmap(list_widget: QListWidget) -> QDrag:
    # this a good example of how to set the pixmap for the item being
    # dragged in a QListWidget using custom widgets.
    # We may want to reuse this in the future for QtLayerList
    drag = QDrag(list_widget)
    drag.setMimeData(list_widget.mimeData(list_widget.selectedItems()))
    size = list_widget.viewport().visibleRegion().boundingRect().size()
    pixmap = QPixmap(size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    for index in list_widget.selectedIndexes():
        rect = list_widget.visualRect(index)
        painter.drawPixmap(rect, list_widget.viewport().grab(rect))
    painter.end()
    drag.setPixmap(pixmap)
    drag.setHotSpot(list_widget.viewport().mapFromGlobal(QCursor.pos()))
    return drag
