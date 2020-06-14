"""Provides a QtPluginSorter that allows the user to change plugin call order.
"""
import re
from typing import List, Optional, Union

from qtpy.QtCore import QEvent, Qt, Signal, Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from napari_plugin_engine import HookImplementation, HookCaller, PluginManager
from .utils import drag_with_pixmap


def rst2html(text):
    def ref(match):
        _text = match.groups()[0].split()[0]
        if _text.startswith("~"):
            _text = _text.split(".")[-1]
        return f'``{_text}``'

    def link(match):
        _text, _link = match.groups()[0].split('<')
        return f'<a href="{_link.rstrip(">")}">{_text.strip()}</a>'

    text = re.sub(r'\*\*([^\*]+)\*\*', '<strong>\\1</strong>', text)
    text = re.sub(r'\*([^\*]+)\*', '<em>\\1</em>', text)
    text = re.sub(r':[a-z]+:`([^`]+)`', ref, text, re.DOTALL)
    text = re.sub(r'`([^`]+)`_', link, text, re.DOTALL)
    text = re.sub(r'``([^`]+)``', '<code>\\1</code>', text)
    return text.replace("\n", "<br>")


class ImplementationListItem(QFrame):
    """A Widget to render each hook implementation item in a ListWidget.

    Parameters
    ----------
    item : QListWidgetItem
        An item instance from a QListWidget. This will most likely come from
        :meth:`QtHookImplementationListWidget.add_hook_implementation_to_list`.
    parent : QWidget, optional
        The parent widget, by default None

    Attributes
    ----------
    plugin_name_label : QLabel
        The name of the plugin providing the hook implementation.
    enabled_checkbox : QCheckBox
        Checkbox to set the ``enabled`` status of the corresponding hook
        implementation.
    opacity : QGraphicsOpacityEffect
        The opacity of the whole widget.  When self.enabled_checkbox is
        unchecked, the opacity of the item is decreased.
    """

    def __init__(self, item: QListWidgetItem, parent: QWidget = None):
        super().__init__(parent)
        self.setToolTip("Click and drag to change call order")
        self.item = item
        self.opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.position_label = QLabel()
        self.update_position_label()

        self.plugin_name_label = QLabel(item.hook_implementation.plugin_name)
        self.enabled_checkbox = QCheckBox(self)
        self.enabled_checkbox.setToolTip("Uncheck to disable this plugin")
        self.enabled_checkbox.stateChanged.connect(self._set_enabled)
        self.enabled_checkbox.setChecked(
            getattr(item.hook_implementation, 'enabled', True)
        )
        layout.addWidget(self.position_label)
        layout.addWidget(self.enabled_checkbox)
        layout.addWidget(self.plugin_name_label)
        layout.setStretch(2, 1)
        layout.setContentsMargins(0, 0, 0, 0)

    def _set_enabled(self, state: Union[bool, int]):
        """Set the enabled state of this hook implementation to ``state``."""
        self.item.hook_implementation.enabled = bool(state)
        self.opacity.setOpacity(1 if state else 0.5)

    def update_position_label(self, order=None):
        """Update the label showing the position of this item in the list.

        Parameters
        ----------
        order : list, optional
            A HookOrderType list ... unused by this function, but here for ease
            of signal connection, by default None.
        """
        position = self.item.listWidget().indexFromItem(self.item).row() + 1
        self.position_label.setText(str(position))


class QtHookImplementationListWidget(QListWidget):
    """A ListWidget to display & sort the call order of a hook implementation.

    This class will usually be instantiated by a
    :class:`~napari._qt.qt_plugin_sorter.QtPluginSorter`.  Each item in the list
    will be rendered as a :class:`ImplementationListItem`.

    Parameters
    ----------
    parent : QWidget, optional
        Optional parent widget, by default None
    hook : HookCaller, optional
        The ``HookCaller`` for which to show implementations. by default None
        (i.e. no hooks shown)

    Attributes
    ----------
    hook_caller : HookCaller or None
        The current ``HookCaller`` instance being shown in the list.
    """

    order_changed = Signal(list)  # emitted when the user changes the order.

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        hook_caller: Optional[HookCaller] = None,
    ):
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
        self.hook_caller: Optional[HookCaller] = None
        self.set_hook_caller(hook_caller)

    def set_hook_caller(self, hook_caller: Optional[HookCaller]):
        """Set the list widget to show hook implementations for ``hook_caller``.

        Parameters
        ----------
        hook : HookCaller, optional
            A ``HookCaller`` for which to show implementations. by default None
            (i.e. no hooks shown)
        """
        self.clear()
        self.hook_caller = hook_caller
        if not hook_caller:
            return

        # _nonwrappers returns hook implementations in REVERSE call order
        # so we reverse them here to show them in the list in the order in
        # which they get called.
        for hook_implementation in reversed(hook_caller._nonwrappers):
            self.append_hook_implementation(hook_implementation)

    def append_hook_implementation(
        self, hook_implementation: HookImplementation
    ):
        """Add a list item for ``hook_implementation`` with a custom widget.

        Parameters
        ----------
        hook_implementation : HookImplementation
            The hook implementation object to add to the list.
        """
        item = QListWidgetItem(parent=self)
        item.hook_implementation = hook_implementation
        self.addItem(item)
        widg = ImplementationListItem(item, parent=self)
        item.setSizeHint(widg.sizeHint())
        self.order_changed.connect(widg.update_position_label)
        self.setItemWidget(item, widg)

    def dropEvent(self, event: QEvent):
        """Triggered when the user moves & drops one of the items in the list.

        Parameters
        ----------
        event : QEvent
            The event that triggered the dropEvent.
        """
        super().dropEvent(event)
        order = [self.item(r).hook_implementation for r in range(self.count())]
        self.order_changed.emit(order)

    def startDrag(self, supportedActions: Qt.DropActions):
        drag = drag_with_pixmap(self)
        drag.exec_(supportedActions, Qt.MoveAction)

    @Slot(list)
    def permute_hook(self, order: List[HookImplementation]):
        """Rearrage the call order of the hooks for the current hook impl.

        Parameters
        ----------
        order : list
            A list of str, hook_implementation, or module_or_class, with the
            desired CALL ORDER of the hook implementations.
        """
        if not self.hook_caller:
            return
        self.hook_caller.bring_to_front(order)


class QtPluginSorter(QWidget):
    """Dialog that allows a user to change the call order of plugin hooks.

    A main QComboBox lets the user pick which hook specification they would
    like to reorder.  Then a :class:`QtHookImplementationListWidget` shows the
    current call order for all implementations of the current hook
    specification.  The user may then reorder them, or disable them by checking
    the checkbox next to each hook implementation name.

    Parameters
    ----------
    plugin_manager : PluginManager, optional
        An instance of a PluginManager. by default, the main
        :class:`~napari.plugins.manager.PluginManager` instance
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

    Attributes
    ----------
    hook_combo_box : QComboBox
        A dropdown menu to select the current hook.
    hook_list : QtHookImplementationListWidget
        The list widget that displays (and allows sorting of) all of the hook
        implementations for the currently selected hook.
    """

    NULL_OPTION = 'select hook... '

    def __init__(
        self,
        plugin_manager: PluginManager = napari_plugin_manager,
        *,
        parent: Optional[QWidget] = None,
        initial_hook: Optional[str] = None,
        firstresult_only: bool = True,
    ):
        super().__init__(parent)
        self.plugin_manager = plugin_manager
        self.hook_combo_box = QComboBox()
        self.hook_combo_box.addItem(self.NULL_OPTION, None)

        # populate comboBox with all of the hooks known by the plugin manager
        for name, hook_caller in plugin_manager.hooks.items():
            if firstresult_only:
                # if the firstresult_only option is set
                # we only want to include hook_specifications that declare the
                # "firstresult" option as True.
                if not hook_caller.spec.opts.get('firstresult', False):
                    continue
            self.hook_combo_box.addItem(
                name.replace("napari_", ""), hook_caller
            )
        self.hook_combo_box.setToolTip(
            "select the hook specification to reorder"
        )
        self.hook_combo_box.currentIndexChanged.connect(self._on_hook_change)
        self.hook_list = QtHookImplementationListWidget(parent=self)

        title = QLabel('Plugin Sorter')
        title.setObjectName("h3")

        instructions = QLabel(
            'Select a hook to rearrange, then drag and '
            'drop plugins into the desired call order.\n\n'
            'Disable plugins for a specific hook by unchecking their checkbox.'
        )
        instructions.setWordWrap(True)

        self.docstring = QLabel(self)
        self.info = QLabel(self)
        self.info.setObjectName("info_icon")
        doc_lay = QHBoxLayout()
        doc_lay.addWidget(self.docstring)
        doc_lay.setStretch(0, 1)
        doc_lay.addWidget(self.info)

        self.docstring.setWordWrap(True)
        self.docstring.setObjectName('small_text')
        self.info.hide()
        self.docstring.hide()

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(instructions)
        layout.addWidget(self.hook_combo_box)
        layout.addLayout(doc_lay)
        layout.addWidget(self.hook_list)
        layout.setContentsMargins(2, 0, 0, 0)

        if initial_hook is not None:
            self.set_hookname(initial_hook)

    def set_hookname(self, hook: str):
        """Change the hook specification shown in the list widget.

        Parameters
        ----------
        hook : str
            Name of the new hook specification to show.
        """
        self.hook_combo_box.setCurrentText(hook.replace("napari_", ''))

    def _on_hook_change(self, index):
        hook_caller = self.hook_combo_box.currentData()
        self.hook_list.set_hook_caller(hook_caller)

        if hook_caller:
            doc = hook_caller.spec.function.__doc__
            html = rst2html(doc.split("Parameters")[0].strip())
            summary, fulldoc = html.split('<br>', 1)
            while fulldoc.startswith('<br>'):
                fulldoc = fulldoc[4:]
            self.docstring.setText(summary.strip())
            self.docstring.show()
            self.info.show()
            self.info.setToolTip(fulldoc.strip())
        else:
            self.docstring.hide()
            self.info.hide()
            self.docstring.setToolTip('')

    def refresh(self):
        self._on_hook_change(self.hook_combo_box.currentIndex())
