from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QStackedWidget,
    QToolBar,
    QWidget,
)

from .qt_viewer_buttons import QtStateButton


class QtSideBarButton(QtStateButton):
    pass


class QtSideBarWidget(QWidget):
    def __init__(self, parent: QWidget = None, name: str = None):
        super().__init__(parent)

        self._stack = QStackedWidget(self)
        self._buttons = QToolBar(self)
        self._group = QButtonGroup(self)
        self._button_list = []

        # Widget setup
        self.setObjectName(name)
        self._buttons.setOrientation(Qt.Vertical)
        self._buttons.setVisible(False)
        self._stack.setVisible(False)
        self._group.setExclusive(False)
        self._group.buttonToggled.connect(self._toggle_widget)

        self._layout = QHBoxLayout()
        if name == "left":
            self._layout.addWidget(self._buttons)
            self._layout.addWidget(self._stack)
        else:
            self._layout.addWidget(self._stack)
            self._layout.addWidget(self._buttons)

        self._stack.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._buttons.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

    def add_widget(self, name, text, widget, location="top"):
        """"""
        button = QtSideBarButton(name, self, "_test", None)
        button.setToolTip(text)
        button._widget = widget
        self._button_list.append(button)
        self._buttons.addWidget(button)
        self._group.addButton(button)
        self._buttons.setVisible(True)
        self._stack.addWidget(widget)

    def _toggle_widget(self, button, value):
        """"""
        for btn in self._button_list:
            if btn != button:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)

        button.setChecked(value)
        widget = button._widget
        if value:
            self._stack.setCurrentWidget(widget)

        self._stack.setVisible(value)


class QtSideBar(QToolBar):
    """"""

    def __init__(self, parent: QWidget = None, name: str = None):
        super().__init__(parent)

        self._widget = QtSideBarWidget(self, name=name)

        self.setMovable(False)
        self.setObjectName(name)
        self.addWidget(self._widget)
        self.setContentsMargins(0, 0, 0, 0)

    def add_widget(self, name, text, widget, location="top"):
        self._widget.add_widget(name, text, widget, location=location)
