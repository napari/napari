import json
from enum import EnumMeta
from typing import TYPE_CHECKING, Tuple

from qtpy.QtCore import QSize, Signal
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
)

from ..._vendor.qt_json_builder.qt_jsonschema_form import WidgetBuilder
from ...settings import get_settings
from ...utils.translations import trans
from .qt_message_dialogs import ResetNapariInfoDialog

if TYPE_CHECKING:
    from pydantic.fields import ModelField


class PreferencesDialog(QDialog):
    """Preferences Dialog for Napari user settings."""

    ui_schema = {
        "call_order": {"ui:widget": "plugins"},
        "highlight_thickness": {"ui:widget": "highlight"},
        "shortcuts": {"ui:widget": "shortcuts"},
    }

    resized = Signal(QSize)
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans._("Preferences"))

        self._stack = QStackedWidget(self)
        self._list = QListWidget(self)
        self._list.setObjectName("Preferences")
        self._list.currentRowChanged.connect(self._stack.setCurrentIndex)

        # Set up buttons
        self._button_cancel = QPushButton(trans._("Cancel"))
        self._button_cancel.clicked.connect(self.reject)
        self._button_ok = QPushButton(trans._("OK"))
        self._button_ok.clicked.connect(self.accept)
        self._button_ok.setDefault(True)
        self._default_restore = QPushButton(trans._("Restore defaults"))
        self._default_restore.clicked.connect(self.restore_defaults)

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self._list)
        left_layout.addStretch()
        left_layout.addWidget(self._default_restore)
        left_layout.addWidget(self._button_cancel)
        left_layout.addWidget(self._button_ok)

        self.setLayout(QHBoxLayout())
        self.layout().addLayout(left_layout, 1)
        self.layout().addWidget(self._stack, 3)

        # Build dialog from settings
        self._starting_values = {}
        self._rebuild_dialog()

    def _restart_dialog(self):
        """Displays the dialog informing user a restart is required."""
        text_str = trans._(
            "napari requires a restart for image rendering changes to apply."
        )
        widget = ResetNapariInfoDialog(parent=self, text=text_str)
        widget.exec_()

    def resizeEvent(self, event):
        """Override to emit signal."""
        self.resized.emit(event.size())
        super().resizeEvent(event)

    def _rebuild_dialog(self):
        """Removes settings not to be exposed to user and creates dialog pages."""

        settings = get_settings()
        self._starting_values = settings.dict()

        self._list.clear()
        while self._stack.count():
            self._stack.removeWidget(self._stack.currentWidget())

        # Because there are multiple pages, need to keep a dictionary of values dicts.
        # One set of keywords are for each page, then in each entry for a page, there are dicts
        # of setting and its value.
        self._values_orig_dict = {}
        self._values_dict = {}
        self._setting_changed_dict = {}

        for page, field in settings.__fields__.items():
            schema, values, properties = self._get_page_dict(field)

            self._setting_changed_dict[page] = {}
            self._values_orig_dict[page] = values
            self._values_dict[page] = values

            # Only add pages if there are any properties to add.
            if properties:
                name = field.field_info.title or field.name
                widget = self._build_page_dialog(schema, values, name)
                self._list.addItem(name)
                self._stack.addWidget(widget)
        self._list.setCurrentRow(0)

    def _get_page_dict(self, field: 'ModelField') -> Tuple[dict, dict, dict]:
        """Provides the schema, set of values for each setting, and the
        properties for each setting."""

        schema = json.loads(field.type_.schema_json())

        # find enums:
        for name, subfield in field.type_.__fields__.items():
            if isinstance(subfield.type_, EnumMeta):
                enums = [s.value for s in subfield.type_]  # type: ignore
                schema["properties"][name]["enum"] = enums
                schema["properties"][name]["type"] = "string"

        # Need to remove certain properties that will not be displayed on the GUI
        properties = schema.pop('properties')
        setting = getattr(get_settings(), field.name)
        with setting.enums_as_values():
            values = setting.dict()
        napari_config = getattr(setting, "NapariConfig", None)
        for val in napari_config.preferences_exclude:
            properties.pop(val)
            values.pop(val)

        schema['properties'] = properties

        return schema, values, properties

    def _build_page_dialog(self, schema: dict, values: dict, name: str):
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

        # TODO: this shouldn't live here... if there is a coupling/dependency
        # between these settings, it should be declared in the settings schema
        # need to disable async if octree is enabled.
        if name.lower() == 'experimental' and values['octree'] is True:
            form = self._disable_async(form, values)

        form.widget.on_changed.connect(lambda *a: print("changed", a))
        form.widget.on_changed.connect(
            lambda d: self.check_differences(
                d, self._values_dict[name.lower()]
            )
        )

        return form

    def restore_defaults(self):
        """Launches dialog to confirm restore settings choice."""
        response = QMessageBox.question(
            self,
            trans._("Restore Settings"),
            trans._("Are you sure you want to restore default settings?"),
            QMessageBox.RestoreDefaults | QMessageBox.Cancel,
            QMessageBox.RestoreDefaults,
        )
        if response == QMessageBox.RestoreDefaults:
            get_settings().reset()
            self._rebuild_dialog()  # TODO: do we need this?

    def _disable_async(self, form, values, disable=True, state=True):
        """Disable async if octree is True."""
        # need to make sure that if async_ is an environment setting, that we don't
        # enable it here.

        env_settings = get_settings().env_settings().get('experimental', {})
        if env_settings.get('async_') not in (None, '0'):
            disable = True

        idx = list(values.keys()).index('async_')
        form_layout = form.widget.layout()
        widget = form_layout.itemAt(idx, form_layout.FieldRole).widget()
        widget.opacity.setOpacity(0.3 if disable else 1)
        widget.setDisabled(disable)

        return form

    def reject(self):
        """Restores the settings in place when dialog was launched."""
        get_settings().update(self._starting_values)
        super().reject()

        # # Need to check differences for each page.
        # settings = get_settings()
        # for n in range(self._stack.count()):
        #     # Must set the current row so that the proper list is updated
        #     # in check differences.
        #     self._list.setCurrentRow(n)
        #     page = self._list.currentItem().text().split(" ")[0].lower()
        #     # get new values for settings.  If they were changed from values at beginning
        #     # of preference dialog session, change them back.
        #     # Using the settings value seems to be the best way to get the checkboxes right
        #     # on the plugin call order widget.
        #     field = settings.__fields__[page]
        #     _, new_values, _ = self._get_page_dict(field)
        #     self.check_differences(self._values_orig_dict[page], new_values)

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
            elif setting_name in self._setting_changed_dict[page]:
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
        settings = get_settings()
        page = self._list.currentItem().text().split(" ")[0].lower()
        self._values_changed(page, new_dict, old_dict)
        different_values = self._setting_changed_dict[page]

        if len(different_values) > 0:
            # change the values in settings
            for setting_name, value in different_values.items():
                try:
                    setattr(getattr(settings, page), setting_name, value)
                    self._values_dict[page] = new_dict

                    if page == 'experimental':

                        if setting_name == 'octree':

                            # disable/enable async checkbox
                            widget = self._stack.currentWidget()
                            cstate = value is True
                            self._disable_async(
                                widget, new_dict, disable=cstate
                            )

                            # need to inform user that napari restart needed.
                            self._restart_dialog()

                        elif setting_name == 'async_':
                            # need to inform user that napari restart needed.
                            self._restart_dialog()

                except:  # noqa: E722
                    continue
