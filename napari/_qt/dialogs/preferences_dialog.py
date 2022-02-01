import json
from enum import EnumMeta
from typing import TYPE_CHECKING, Tuple, cast

from pydantic.main import BaseModel
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
)

from ...utils.translations import trans

if TYPE_CHECKING:
    from pydantic.fields import ModelField
    from qtpy.QtGui import QCloseEvent, QKeyEvent


class PreferencesDialog(QDialog):
    """Preferences Dialog for Napari user settings."""

    ui_schema = {
        "call_order": {"ui:widget": "plugins"},
        "highlight_thickness": {"ui:widget": "highlight"},
        "shortcuts": {"ui:widget": "shortcuts"},
        "extension2reader": {"ui:widget": "extension2reader"},
    }

    resized = Signal(QSize)

    def __init__(self, parent=None):
        from ...settings import get_settings

        super().__init__(parent)
        self.setWindowTitle(trans._("Preferences"))

        self._settings = get_settings()
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
        self._button_restore = QPushButton(trans._("Restore defaults"))
        self._button_restore.clicked.connect(self._restore_default_dialog)

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self._list)
        left_layout.addStretch()
        left_layout.addWidget(self._button_restore)
        left_layout.addWidget(self._button_cancel)
        left_layout.addWidget(self._button_ok)

        self.setLayout(QHBoxLayout())
        self.layout().addLayout(left_layout, 1)
        self.layout().addWidget(self._stack, 3)

        # Build dialog from settings
        self._rebuild_dialog()

    def keyPressEvent(self, e: 'QKeyEvent'):
        if e.key() == Qt.Key_Escape:
            # escape key should just close the window
            # which implies "accept"
            e.accept()
            self.accept()
            return
        super().keyPressEvent(e)

    def resizeEvent(self, event):
        """Override to emit signal."""
        self.resized.emit(event.size())
        super().resizeEvent(event)

    def _rebuild_dialog(self):
        """Removes settings not to be exposed to user and creates dialog pages."""
        # FIXME: this dialog should not need to know about the plugin manager
        from ...plugins import plugin_manager

        self._starting_pm_order = plugin_manager.call_order()
        self._starting_values = self._settings.dict(exclude={'schema_version'})

        self._list.clear()
        while self._stack.count():
            self._stack.removeWidget(self._stack.currentWidget())

        for field in self._settings.__fields__.values():
            if isinstance(field.type_, type) and issubclass(
                field.type_, BaseModel
            ):
                self._add_page(field)

        self._list.setCurrentRow(0)

    def _add_page(self, field: 'ModelField'):
        """Builds the preferences widget using the json schema builder.

        Parameters
        ----------
        field : ModelField
            subfield for which to create a page.
        """
        from ..._vendor.qt_json_builder.qt_jsonschema_form import WidgetBuilder

        schema, values = self._get_page_dict(field)
        name = field.field_info.title or field.name

        form = WidgetBuilder().create_form(schema, self.ui_schema)
        # set state values for widget
        form.widget.state = values
        # make settings follow state of the form widget
        form.widget.on_changed.connect(
            lambda d: getattr(self._settings, name.lower()).update(d)
        )

        # need to disable async if octree is enabled.
        # TODO: this shouldn't live here... if there is a coupling/dependency
        # between these settings, it should be declared in the settings schema
        if (
            name.lower() == 'experimental'
            and values['octree']
            and self._settings.env_settings()
            .get('experimental', {})
            .get('async_')
            not in (None, '0')
        ):
            form_layout = form.widget.layout()
            for i in range(form_layout.count()):
                wdg = form_layout.itemAt(i, form_layout.FieldRole).widget()
                if getattr(wdg, '_name') == 'async_':
                    wdg.opacity.setOpacity(0.3)
                    wdg.setDisabled(True)
                    break

        self._list.addItem(field.field_info.title or field.name)
        self._stack.addWidget(form)

    def _get_page_dict(self, field: 'ModelField') -> Tuple[dict, dict, dict]:
        """Provides the schema, set of values for each setting, and the
        properties for each setting."""
        ftype = cast('BaseModel', field.type_)
        schema = json.loads(ftype.schema_json())

        # find enums:
        for name, subfield in ftype.__fields__.items():
            if isinstance(subfield.type_, EnumMeta):
                enums = [s.value for s in subfield.type_]  # type: ignore
                schema["properties"][name]["enum"] = enums
                schema["properties"][name]["type"] = "string"

        # Need to remove certain properties that will not be displayed on the GUI
        setting = getattr(self._settings, field.name)
        with setting.enums_as_values():
            values = setting.dict()
        napari_config = getattr(setting, "NapariConfig", None)
        if hasattr(napari_config, 'preferences_exclude'):
            for val in napari_config.preferences_exclude:
                schema['properties'].pop(val, None)
                values.pop(val, None)

        return schema, values

    def _restore_default_dialog(self):
        """Launches dialog to confirm restore settings choice."""
        response = QMessageBox.question(
            self,
            trans._("Restore Settings"),
            trans._("Are you sure you want to restore default settings?"),
            QMessageBox.RestoreDefaults | QMessageBox.Cancel,
            QMessageBox.RestoreDefaults,
        )
        if response == QMessageBox.RestoreDefaults:
            self._settings.reset()
            self._rebuild_dialog()  # TODO: do we need this?

    def _restart_required_dialog(self):
        """Displays the dialog informing user a restart is required."""
        QMessageBox.information(
            self,
            trans._("Restart required"),
            trans._(
                "A restart is required for some new settings to have an effect."
            ),
        )

    def closeEvent(self, event: 'QCloseEvent') -> None:
        event.accept()
        self.accept()

    def reject(self):
        """Restores the settings in place when dialog was launched."""
        self._settings.update(self._starting_values)

        # FIXME: this dialog should not need to know about the plugin manager
        if self._starting_pm_order:
            from ...plugins import plugin_manager

            plugin_manager.set_call_order(self._starting_pm_order)
        super().reject()
