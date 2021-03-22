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

    resized = Signal(QSize)

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

    def resizeEvent(self, event):
        """Override to emit signal."""
        self.resized.emit(event.size())
        super().resizeEvent(event)

    def make_dialog(self):
        """Removes settings not to be exposed to user and creates dialog pages."""
        # Because there are multiple pages, need to keep a list of values sets.
        self._values_orig_set_list = []
        self._values_set_list = []
        for _key, setting in SETTINGS.schemas().items():
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
            self._values_orig_set_list.append(set(values.items()))
            self._values_set_list.append(set(values.items()))

            # Only add pages if there are any properties to add.
            if properties:
                self.add_page(schema, values)

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
            # Must set the current row so that the proper set list is updated
            # in check differences.
            self._list.setCurrentRow(n)
            self.check_differences(
                self._values_orig_set_list[n],
                self._values_set_list[n],
            )
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
        form = builder.create_form(schema, {})
        # set state values for widget
        form.widget.state = values
        form.widget.on_changed.connect(
            lambda d: self.check_differences(
                set(d.items()),
                self._values_set_list[self._list.currentIndex().row()],
            )
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

        page = self._list.currentItem().text().split(" ")[0].lower()
        different_values = list(new_set - values_set)

        if len(different_values) > 0:
            # change the values in SETTINGS
            for val in different_values:
                try:
                    setattr(SETTINGS._settings[page], val[0], val[1])
                    self._values_set_list[
                        self._list.currentIndex().row()
                    ] = new_set
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
