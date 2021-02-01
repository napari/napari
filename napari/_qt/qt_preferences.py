"""
"""

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
)


class PreferencesDialog(QDialog):
    sig_resized = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)

        self._list = QListWidget(self)
        self._stack = QStackedWidget(self)

        # Use a Buttons group (forgot the name)
        self._button_cancel = QPushButton("Ok")
        self._button_ok = QPushButton("Cancel")

        # Setup
        self.setWindowTitle("Preferences")

        # Layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self._list)
        main_layout.addWidget(self._stack)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self._button_cancel)
        buttons_layout.addWidget(self._button_ok)

        layout = QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

        # Signals
        self._list.currentRowChanged.connect(
            lambda index: self._stack.setCurrentIndex(index)
        )
        self._button_ok.clicked.connect(lambda: print("ok"))
        self._button_cancel.clicked.connect(lambda: print("cancel"))

    def add_page(self, schema):
        """"""
        # widget = build_form_from_schema(schema)
        # self._list.addItem(schema["title"])
        # self._stack.addWidget(widget)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.sig_resized.emit(event)
