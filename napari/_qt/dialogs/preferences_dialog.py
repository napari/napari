from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from napari._vendor.qt_json_builder.qt_jsonschema_form import WidgetBuilder

from ...utils.settings import SETTINGS
from ...utils.translations import translator

trans = translator.load()


class PreferencesDialog(QDialog):
    """Preferences Dialog for Napari user settings."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._list = QListWidget(self)
        self._stack = QStackedWidget(self)

        # Set up buttons
        self._button_cancel = QPushButton(trans._("Cancel"))
        self._button_ok = QPushButton(trans._("OK"))
        self._default_restore = QPushButton(trans._("Restore defaults"))

        # Setup
        self.setWindowTitle(trans._("Preferences"))

        # Layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self._list)
        main_layout.addWidget(self._stack)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self._button_cancel)
        buttons_layout.addWidget(self._button_ok)

        layout = QVBoxLayout()
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
        """Launches dialog to confirm restore settings choice."""

        widget = ConfirmDialog(
            parent=self,
            text=trans._("Are you sure you want to restore default settings?"),
        )
        widget.show()

    def on_click_ok(self):
        """Keeps the selected preferences saved to SETTINGS."""
        self.close()

    def on_click_cancel(self):
        """Restores the settings in place when dialog was launched."""
        self.check_differences(self._values_orig_set, self._values_set)
        self.close()

    def add_page(self, schema, values):
        """Creates a new page for each section in preferences.

        Parameters
        ----------
        schema : dict
            Json schema including all information to build each page in the
            preferences dialog.
        values : dict
            Dictionary of current values set in preferences.

        """
        widget = self.get_preferences_dialog(schema, values)
        self._list.addItem(schema["title"])
        self._stack.addWidget(widget)

    def get_preferences_dialog(self, schema, values):
        """Builds the preferences widget using the json schema builder.

        Parameters
        ----------
        schema : dict
            Json schema including all information to build each page in the
            preferences dialog.
        values : dict
            Dictionary of current values set in preferences.

        """
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
        """Changes settings in settings manager with changes from dialog.

        Parameters
        ----------
        new_set : set
            The set of new values, with tuples of key value pairs for each
            setting.
        values_set : set
            The old set of values.
        """

        different_values = list(new_set - values_set)

        if len(different_values) > 0:
            # change the values in SETTINGS
            for val in different_values:
                # to do -- reference proper page -- its hard coded for just 'application' right now
                try:
                    # need to validate so a wrong value is not saved at all...
                    setattr(SETTINGS._settings['application'], val[0], val[1])
                    self._values_set = new_set
                except:  # noqa: E722
                    continue


class ConfirmDialog(QDialog):
    """Dialog to confirms a user's choice to restore default settings."""

    def __init__(
        self,
        parent: QWidget = None,
        text: str = "",
    ):
        super().__init__(parent)

        # Set up components
        self._question = QLabel(self)
        self._button_restore = QPushButton(trans._("Restore"))
        self._button_cancel = QPushButton(trans._("Cancel"))

        # Widget set up
        self._question.setText(text)

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self._button_cancel)
        button_layout.addWidget(self._button_restore)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._question)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Signals
        self._button_cancel.clicked.connect(self.on_click_cancel)
        self._button_restore.clicked.connect(self.on_click_restore)

    def on_click_cancel(self):
        """Do not restore defaults and close window."""
        self.close()

    def on_click_restore(self):
        """Restore defaults and close window."""
        SETTINGS.reset()
        self.close()
