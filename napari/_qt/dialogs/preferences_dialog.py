import json

from qtpy.QtCore import QSize, Signal
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

from ..._vendor.qt_json_builder.qt_jsonschema_form import WidgetBuilder
from ...utils.settings import SETTINGS
from ...utils.translations import trans


class PreferencesDialog(QDialog):
    """Preferences Dialog for Napari user settings."""

    ui_schema = {
        "call_order": {"ui:widget": "plugins"},
        "highlight_thickness": {"ui:widget": "highlight"},
    }

    resized = Signal(QSize)
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._list = QListWidget(self)
        self._stack = QStackedWidget(self)

        self._list.setObjectName("Preferences")

        # Set up buttons
        self._button_cancel = QPushButton(trans._("Cancel"))
        self._button_ok = QPushButton(trans._("OK"))
        self._default_restore = QPushButton(trans._("Restore defaults"))

        # Setup
        self.setWindowTitle(trans._("Preferences"))

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self._list)
        left_layout.addStretch()
        left_layout.addWidget(self._default_restore)
        left_layout.addWidget(self._button_cancel)
        left_layout.addWidget(self._button_ok)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self._stack, 3)

        self.setLayout(main_layout)

        # Signals

        self._list.currentRowChanged.connect(
            lambda index: self._stack.setCurrentIndex(index)
        )
        self._button_cancel.clicked.connect(self.on_click_cancel)
        self._button_ok.clicked.connect(self.on_click_ok)
        self._default_restore.clicked.connect(self.restore_defaults)

        # Make widget

        self.make_dialog()
        self._list.setCurrentRow(0)

    def closeEvent(self, event):
        """Override to emit signal."""
        self.closed.emit()
        super().closeEvent(event)

    def reject(self):
        """Override to handle Escape."""
        super().reject()
        self.close()

    def resizeEvent(self, event):
        """Override to emit signal."""
        self.resized.emit(event.size())
        super().resizeEvent(event)

    def make_dialog(self):
        """Removes settings not to be exposed to user and creates dialog pages."""

        # Because there are multiple pages, need to keep a dictionary of values dicts.
        # One set of keywords are for each page, then in each entry for a page, there are dicts
        # of setting and its value.

        self._values_orig_dict = {}
        self._values_dict = {}
        self._setting_changed_dict = {}

        for page, setting in SETTINGS.schemas().items():
            schema, values, properties = self.get_page_dict(setting)

            self._setting_changed_dict[page] = {}
            self._values_orig_dict[page] = values
            self._values_dict[page] = values

            # Only add pages if there are any properties to add.
            if properties:
                self.add_page(schema, values)

    def get_page_dict(self, setting):
        """Provides the schema, set of values for each setting, and the properties
        for each setting.

        Parameters
        ----------
        setting : dict
            Dictionary of settings for a page within the settings manager.

        Returns
        -------
        schema : dict
            Json schema of the setting page.
        values : dict
            Dictionary of values currently set for each parameter in the settings.
        properties : dict
            Dictionary of properties within the json schema.

        """

        schema = json.loads(setting['json_schema'])
        # Need to remove certain properties that will not be displayed on the GUI
        properties = schema.pop('properties')
        model = setting['model']
        values = model.dict()
        napari_config = getattr(model, "NapariConfig", None)
        if napari_config is not None:
            for val in napari_config.preferences_exclude:
                properties.pop(val)
                values.pop(val)

        schema['properties'] = properties

        return schema, values, properties

    def restore_defaults(self):
        """Launches dialog to confirm restore settings choice."""

        widget = ConfirmDialog(
            parent=self,
            text=trans._("Are you sure you want to restore default settings?"),
        )
        widget.valueChanged.connect(self._reset_widgets)
        widget.exec_()

    def _reset_widgets(self):
        """Deletes the widgets and rebuilds with defaults."""
        self.close()
        self._list.clear()

        for n in range(self._stack.count()):
            widget = self._stack.removeWidget(self._stack.currentWidget())
            del widget

        self.make_dialog()
        self._list.setCurrentRow(0)
        self.show()

    def on_click_ok(self):
        """Keeps the selected preferences saved to SETTINGS."""
        self.close()

    def on_click_cancel(self):
        """Restores the settings in place when dialog was launched."""
        # Need to check differences for each page.
        for n in range(self._stack.count()):
            # Must set the current row so that the proper list is updated
            # in check differences.
            self._list.setCurrentRow(n)
            page = self._list.currentItem().text().split(" ")[0].lower()
            # get new values for settings.  If they were changed from values at beginning
            # of preference dialog session, change them back.
            # Using the settings value seems to be the best way to get the checkboxes right
            # on the plugin call order widget.
            setting = SETTINGS.schemas()[page]
            schema, new_values, properties = self.get_page_dict(setting)
            self.check_differences(self._values_orig_dict[page], new_values)

        self._list.setCurrentRow(0)
        self.close()

    def add_page(self, schema, values):
        """Creates a new page for each section in dialog.

        Parameters
        ----------
        schema : dict
            Json schema including all information to build each page in the
            preferences dialog.
        values : dict
            Dictionary of current values set in preferences.
        """
        widget = self.build_page_dialog(schema, values)
        self._list.addItem(schema["title"])
        self._stack.addWidget(widget)

    def build_page_dialog(self, schema, values):
        """Builds the preferences widget using the json schema builder.

        Parameters
        ----------
        schema : dict
            Json schema including all information to build each page in the
            preferences dialog.
        values : dict
            Dictionary of current values set in preferences.
        """

        builder = WidgetBuilder()
        form = builder.create_form(schema, self.ui_schema)
        # set state values for widget
        form.widget.state = values
        form.widget.on_changed.connect(
            lambda d: self.check_differences(
                d,
                self._values_dict[schema["title"].lower()],
            )
        )

        return form

    def _values_changed(self, page, new_dict, old_dict):
        """Loops through each setting in a page to determine if it changed.

        Parameters
        ----------
        new_dict : dict
            Dict that has the most recent changes by user. Each key is a setting value
            and each item is the value.
        old_dict : dict
            Dict wtih values set at the begining of preferences dialog session.

        """
        for setting_name, value in new_dict.items():
            if value != old_dict[setting_name]:
                self._setting_changed_dict[page][setting_name] = value
            elif (
                value == old_dict[setting_name]
                and setting_name in self._setting_changed_dict[page]
            ):
                self._setting_changed_dict[page].pop(setting_name)

    def check_differences(self, new_dict, old_dict):
        """Changes settings in settings manager with changes from dialog.

        Parameters
        ----------
        new_dict : dict
            Dict that has the most recent changes by user. Each key is a setting parameter
            and each item is the value.
        old_dict : dict
            Dict wtih values set at the beginning of the preferences dialog session.
        """
        page = self._list.currentItem().text().split(" ")[0].lower()
        self._values_changed(page, new_dict, old_dict)
        different_values = self._setting_changed_dict[page]

        if len(different_values) > 0:
            # change the values in SETTINGS
            for setting_name, value in different_values.items():
                try:
                    setattr(SETTINGS._settings[page], setting_name, value)
                    self._values_dict[page] = new_dict
                except:  # noqa: E722
                    continue


class ConfirmDialog(QDialog):
    """Dialog to confirms a user's choice to restore default settings."""

    valueChanged = Signal(bool)

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
        self.valueChanged.emit(True)
        self.close()
