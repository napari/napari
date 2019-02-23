from PyQt5.QtWidgets import QHBoxLayout, QStackedWidget, QWidget
from PyQt5.QtCore import QSize


class QtPlugins(QStackedWidget):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer

        self.setMouseTracking(True)
        self.empty_widget = QWidget()
        self.addWidget(self.empty_widget)
        self.setCurrentWidget(self.empty_widget)
        self.setFixedWidth(200)

    def _add(self, plugin):
        if plugin._qt is not None:
            self.addWidget(plugin._qt)
            self.setCurrentWidget(plugin._qt)
            sizes = self.viewer._qt.sizes()
            sizes[-1] = 200
            self.viewer._qt.setSizes(sizes)
