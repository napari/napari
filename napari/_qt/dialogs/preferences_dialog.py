from qtpy import QtWidgets

from napari._vendor.qt_json_builder.qt_jsonschema_form import WidgetBuilder

from ...utils.settings import SETTINGS


class PreferencesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._list = QtWidgets.QListWidget(self)
        self._stack = QtWidgets.QStackedWidget(self)

        # Set up buttons
        self._button_cancel = QtWidgets.QPushButton("Cancel")
        self._button_ok = QtWidgets.QPushButton("OK")
        self._default_restore = QtWidgets.QPushButton("Restore defaults")

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
        layout.addWidget(self._default_restore)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

        # Signals

        self._list.currentRowChanged.connect(
            lambda index: self._stack.setCurrentIndex(index)
        )
        self._button_cancel.clicked.connect(self.on_click_cancel)
        self._button_ok.clicked.connect(self.on_click_ok)
        self._default_restore.clicked.connect(self.restore_defaults)

    def restore_defaults(self):
        SETTINGS.reset()
        self.close()

    def on_click_ok(self):
        # will keep these values set in settings.
        self.close()

    def on_click_cancel(self):
        # reset to already saved values
        self.check_differences(self._values_orig_set, self._values_set)
        self.close()

    def add_page(self, schema, ui_schema):

        widget = self.get_preferences_dialog(schema, ui_schema)
        self._list.addItem(schema["title"])
        self._stack.addWidget(widget)

    def get_preferences_dialog(self, schema, values):

        self._values_orig_set = set(values.items())
        self._values_set = set(values.items())

        builder = WidgetBuilder()

        form = builder.create_form(schema, {})
        # set state values for widget
        form.widget.state = values
        form.widget.on_changed.connect(
            lambda d: self.check_differences(set(d.items()), self._values_set)
        )

        return form

    def check_differences(self, new_set, values_set):
        """
        d: the set of new values
        values_set: the set by which to compare
        """
        different_values = list(new_set - values_set)

        if len(different_values) > 0:
            #     # change the values in SETTINGS
            for val in different_values:
                # to do -- reference proper page -- its hard coded for just 'application' right now
                try:
                    # need to validate so a wrong value is not saved at all...
                    setattr(SETTINGS._settings['application'], val[0], val[1])
                    self._values_set = new_set
                except:  # noqa: E722
                    continue
