from PyQt5.QtWidgets import QHBoxLayout, QStackedWidget, QWidget
from PyQt5.QtCore import QSize


class QtControls(QStackedWidget):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer

        self.setMouseTracking(True)
        self.setMinimumSize(QSize(40, 40))
        self.empty_widget = QWidget()
        self.addWidget(self.empty_widget)
        self.display(None)

        self.viewer.layers.changed.added.connect(self._add)
        self.viewer.layers.changed.removed.connect(self._remove)

    def display(self, layer):
        if layer is None or layer._qt_controls is None:
            self.setCurrentWidget(self.empty_widget)
        else:
            self.setCurrentWidget(layer._qt_controls)

    def _add(self, event):
        layer = event.item
        if layer._qt_controls is not None:
            self.addWidget(layer._qt_controls)

    def _remove(self, event):
        layer = event.item
        if layer._qt_controls is not None:
            self.removeWidget(layer._qt_controls)
            layer._qt_controls.deleteLater()
            layer._qt_controls = None
