from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
)
from ..plugins import plugin_manager as napari_plugin_manager
from ..plugins.manager import permute_hookimpls


class QtHookImplItem(QListWidgetItem):
    def __init__(self, hookimpl, parent=None):
        self.plugin_name = hookimpl.plugin_name
        if self.plugin_name == 'builtins':
            self.plugin_name = 'napari built-in'
        super().__init__(self.plugin_name, parent)
        self.hookimpl = hookimpl
        self.setBackground(Qt.black)


class QtHookImplListWidget(QListWidget):
    order_changed = Signal(list)

    def __init__(self, hook=None, parent=None):
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
        self.order_changed.connect(self.transpose_hook)
        self.set_hook(hook)

    def set_hook(self, hook):
        self.clear()
        self.hook = hook
        if not hook:
            return
        for hookimpl in reversed(hook.get_hookimpls()):
            self.addItem(QtHookImplItem(hookimpl))

    def dropEvent(self, event):
        super().dropEvent(event)
        order = [self.item(r).hookimpl for r in range(self.count())]
        self.order_changed.emit(order)

    def transpose_hook(self, order):
        if not self.hook:
            return
        permute_hookimpls(self.hook, order)


class QtPluginSorter(QDialog):
    NULL_OPTION = 'select hook... '

    def __init__(
        self,
        plugin_manager=None,
        parent=None,
        *,
        initial_hook=None,
        firstresult_only=True,
    ):
        plugin_manager = plugin_manager or napari_plugin_manager
        super().__init__(parent)
        self.pm = plugin_manager
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.hookComboBox = QComboBox()
        self.hookComboBox.addItem(self.NULL_OPTION)
        hooks = []
        for name, hook_caller in vars(plugin_manager.hook).items():
            if firstresult_only:
                if not hook_caller.spec.opts.get('firstresult', False):
                    continue
            hooks.append(name)
        self.hookComboBox.addItems(hooks)
        self.hookComboBox.activated[str].connect(self.change_hook)
        self.hookList = QtHookImplListWidget()
        self.layout.addWidget(self.hookComboBox)
        self.layout.addWidget(self.hookList)
        if initial_hook is not None:
            self.hookComboBox.setCurrentText(initial_hook)
            self.change_hook(initial_hook)

    def change_hook(self, hook):
        if hook == self.NULL_OPTION:
            self.hookList.set_hook(None)
        else:
            self.hookList.set_hook(getattr(self.pm.hook, hook))


if __name__ == "__main__":
    import sys
    from qtpy.QtWidgets import QApplication
    from napari.plugins import plugin_manager

    app = QApplication(sys.argv)
    w = QtPluginSorter(plugin_manager)
    w.show()
    sys.exit(app.exec_())
