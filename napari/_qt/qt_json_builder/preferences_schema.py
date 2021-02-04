from json import dumps

from PyQt5 import QtWidgets

from .qt_jsonschema_form import WidgetBuilder


def get_preferences_dialog(schema, ui_schema):
    #     app = QtWidgets.QApplication(sys.argv)

    builder = WidgetBuilder()

    form = builder.create_form(schema, ui_schema)

    form.widget.on_changed.connect(lambda d: print(dumps(d, indent=4)))
    # form.centralWidget().widget.on_changed.connect(
    #     lambda d: print(dumps(d, indent=4))
    # )
    # form.show()
    return form
    # app.exec_()


class PreferencesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._list = QtWidgets.QListWidget(self)
        self._stack = QtWidgets.QStackedWidget(self)

        # Set up buttons
        self._button_cancel = QtWidgets.QPushButton("Cancel")
        self._button_ok = QtWidgets.QPushButton("OK")

        # Setup
        self.setWindowTitle("Preferences")

        # Layout
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self._list)
        main_layout.addWidget(self._stack)

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self._button_cancel)
        buttons_layout.addWidget(self._button_ok)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

        # Signals

        self._list.currentRowChanged.connect(
            lambda index: self._stack.setCurrentIndex(index)
        )
        self._button_cancel.clicked.connect(self.on_click_cancel)
        self._button_ok.clicked.connect(self.on_click_ok)

    def on_click_ok(self):
        print('OK')
        self.close()

    def on_click_cancel(self):
        print('cancel')
        self.close()

    def add_page(self, schema, ui_schema):

        widget = get_preferences_dialog(schema, ui_schema)
        self._list.addItem(schema["title"])
        self._stack.addWidget(widget)
