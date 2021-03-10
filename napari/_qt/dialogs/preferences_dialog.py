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
    """Preference Dialog that allows user to set their preferences for Napari"""

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
        """Launches a separate dialog that asks the user to confirm the restore
        or to cancel.
        """

        widget = ConfirmDialog(
            parent=self,
            text=trans._("Are you sure you want to restore default settings?"),
        )
        widget.show()

    def on_click_ok(self):
        """Keeps the selected preferences saved to SETTINGS."""
        self.close()

    def on_click_cancel(self):
        """Restores the settings in place when the preference dialog was launched."""
        self.check_differences(self._values_orig_set, self._values_set)
        self.close()

    def add_page(self, schema, ui_schema):
        """For each section in settings, a new preferences widget will be added to the dialog
        on a separate page.
        """
        widget = self.get_preferences_dialog(schema, ui_schema)
        self._list.addItem(schema["title"])
        self._stack.addWidget(widget)

    def get_preferences_dialog(self, schema, values):
        """Builds the preferences widget using the json schema builder"""
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
        """Will check the differences in settings from the original values set and the new set.
            For all values that are new, the value in SETTINGS will be updated.

        Parameters
        ----------
        new_set: the set of new values
        values_set: the set by which to compare
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
    """Dialog that confirms a user's choice to restore default settings."""

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
        """Do not restore defaults.  Close window."""
        self.close()

    def on_click_restore(self):
        """Restore defaults.  Close window."""
        SETTINGS.reset()
        self.close()
