from enum import EnumMeta
from typing import TYPE_CHECKING, ClassVar, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
)

from napari._pydantic_util import get_inner_type
from napari.utils.compat import StrEnum
from napari.utils.translations import trans

if TYPE_CHECKING:
    from qtpy.QtGui import QCloseEvent, QKeyEvent


class PreferencesDialog(QDialog):
    """Preferences Dialog for Napari user settings."""

    ui_schema: ClassVar[dict[str, dict[str, str]]] = {
        'highlight': {'ui:widget': 'highlight'},
        'shortcuts': {'ui:widget': 'shortcuts'},
        'extension2reader': {'ui:widget': 'extension2reader'},
        'dask': {'ui:widget': 'horizontal_object'},
        'font_size': {'ui:widget': 'font_size'},
    }

    resized = Signal(QSize)

    def __init__(self, parent=None) -> None:
        from napari.settings import get_settings

        super().__init__(parent)
        self.setWindowTitle(trans._('Preferences'))
        self.setMinimumSize(QSize(1065, 470))

        self._settings = get_settings()
        self._stack = QStackedWidget(self)
        self._list = QListWidget(self)
        self._list.setObjectName('Preferences')
        self._list.currentRowChanged.connect(self._stack.setCurrentIndex)

        # Set up buttons
        self._button_cancel = QPushButton(trans._('Cancel'))
        self._button_cancel.clicked.connect(self.reject)
        self._button_ok = QPushButton(trans._('OK'))
        self._button_ok.clicked.connect(self.accept)
        self._button_ok.setDefault(True)
        self._button_restore = QPushButton(trans._('Restore defaults'))
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
        self.layout().addWidget(self._stack, 4)

        # Build dialog from settings
        self._rebuild_dialog()

    def keyPressEvent(self, e: 'QKeyEvent'):
        if e.key() == Qt.Key.Key_Escape:
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

        self._starting_values = self._settings.dict(exclude={'schema_version'})

        self._list.clear()
        while self._stack.count():
            self._stack.removeWidget(self._stack.currentWidget())

        for field_name, field_info in self._settings.model_fields.items():
            field_type = get_inner_type(field_info.annotation)
            if get_origin(field_type) is None and issubclass(
                field_type, BaseModel
            ):
                self._add_page(field_name, field_info)

        self._list.setCurrentRow(0)

    def _add_page(self, field_name: str, field_info: FieldInfo):
        """Builds the preferences widget using the json schema builder.

        Parameters
        ----------
        field : FieldInfo
            subfield for which to create a page.
        """
        from napari._vendor.qt_json_builder.qt_jsonschema_form import (
            WidgetBuilder,
        )

        schema, values = self._get_page_dict(field_name, field_info)
        name = field_info.title or field_name

        form = WidgetBuilder().create_form(schema, self.ui_schema)
        # set state values for widget
        form.widget.state = values
        # make settings follow state of the form widget
        form.widget.on_changed.connect(
            lambda d: getattr(self._settings, name.lower()).update(d)
        )
        # make widgets follow values of the settings
        settings_category = getattr(self._settings, name.lower())
        excluded = set(
            getattr(
                getattr(settings_category, 'NapariConfig', None),
                'preferences_exclude',
                {},
            )
        )
        nested_settings = ['dask', 'highlight']
        for name_, emitter in settings_category.events.emitters.items():
            if name_ not in excluded and name_ not in nested_settings:
                emitter.connect(update_widget_state(name_, form.widget))
            elif name_ in nested_settings:
                # Needed to handle nested event model settings (i.e `DaskSettings` and `HighlightSettings`)
                for subname_, subemitter in getattr(
                    settings_category, name_
                ).events.emitters.items():
                    subemitter.connect(
                        update_widget_state(
                            subname_, form.widget.widgets[name_]
                        )
                    )

        page_scrollarea = QScrollArea()
        page_scrollarea.setWidgetResizable(True)
        page_scrollarea.setWidget(form)

        self._list.addItem(name)
        self._stack.addWidget(page_scrollarea)

    def _get_page_dict(
        self, field_name: str, field_info: FieldInfo
    ) -> tuple[dict, dict]:
        """Provides the schema, set of values for each setting, and the
        properties for each setting."""
        ftype = field_info.annotation

        # TODO make custom shortcuts dialog to properly capture new
        #      functionality once we switch to app-model's keybinding system
        #      then we can remove the below code used for autogeneration
        if field_name == 'shortcuts':
            # hardcode workaround because pydantic's schema generation
            # does not allow you to specify custom JSON serialization
            schema = {
                'title': 'ShortcutsSettings',
                'type': 'object',
                'properties': {
                    'shortcuts': {
                        'title': ftype.model_fields['shortcuts'].title,
                        'description': ftype.model_fields[
                            'shortcuts'
                        ].description,
                        'type': 'object',
                    }
                },
            }
        else:
            schema = ftype.model_json_schema()

        if field_info.title:
            schema['title'] = field_info.title
        if field_info.description:
            schema['description'] = field_info.description

        # find enums:
        for subfield_name, subfield_info in ftype.model_fields.items():
            sftype = get_inner_type(subfield_info.annotation)
            if isinstance(sftype, EnumMeta):
                enums = [s.value for s in sftype]
                schema['properties'][subfield_name]['enum'] = enums
                schema['properties'][subfield_name]['type'] = 'string'
            if get_origin(sftype) is None and issubclass(sftype, BaseModel):
                local_schema = sftype.model_json_schema()
                schema['properties'][subfield_name]['type'] = 'object'
                schema['properties'][subfield_name]['properties'] = (
                    local_schema['properties']
                )

        # Need to remove certain properties that will not be displayed on the GUI
        setting = getattr(self._settings, field_name)
        with setting.enums_as_values():
            values = setting.model_dump()
        napari_config = getattr(setting, 'NapariConfig', None)
        if hasattr(napari_config, 'preferences_exclude'):
            for val in napari_config.preferences_exclude:
                schema['properties'].pop(val, None)
                values.pop(val, None)

        return schema, values

    def _restore_default_dialog(self):
        """Launches dialog to confirm restore settings choice."""
        prev = QApplication.instance().testAttribute(
            Qt.ApplicationAttribute.AA_DontUseNativeDialogs
        )
        QApplication.instance().setAttribute(
            Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True
        )

        response = QMessageBox.question(
            self,
            trans._('Restore Settings'),
            trans._('Are you sure you want to restore default settings?'),
            QMessageBox.StandardButton.RestoreDefaults
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.RestoreDefaults,
        )
        QApplication.instance().setAttribute(
            Qt.ApplicationAttribute.AA_DontUseNativeDialogs, prev
        )
        if response == QMessageBox.RestoreDefaults:
            self._settings.reset()

    def _restart_required_dialog(self):
        """Displays the dialog informing user a restart is required."""
        QMessageBox.information(
            self,
            trans._('Restart required'),
            trans._(
                'A restart is required for some new settings to have an effect.'
            ),
        )

    def closeEvent(self, event: 'QCloseEvent') -> None:
        event.accept()
        self.accept()

    def accept(self):
        self._settings.save()
        super().accept()

    def reject(self):
        """Restores the settings in place when dialog was launched."""
        self._settings.update(self._starting_values)
        super().reject()


def update_widget_state(name, widget):
    def _update_widget_state(event):
        value = event.value
        if isinstance(value, StrEnum):
            value = value.value
        widget.state = {name: value}

    return _update_widget_state
