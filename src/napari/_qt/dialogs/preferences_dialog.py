from enum import EnumMeta, StrEnum
from typing import TYPE_CHECKING, ClassVar, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from napari._pydantic_util import get_inner_type

if TYPE_CHECKING:
    from qtpy.QtGui import QCloseEvent, QKeyEvent, QResizeEvent


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
        from napari.settings import get_plugin_settings, get_settings

        super().__init__(parent)
        self.setWindowTitle('Preferences')
        self.setMinimumSize(QSize(1065, 470))

        self._settings = get_settings()
        self._plugin_settings = get_plugin_settings()
        self._stack = QStackedWidget(self)
        self._list = QListWidget(self)
        self._list.setObjectName('Preferences')
        self._list.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        # Set up buttons
        self._button_cancel = QPushButton('Cancel')
        self._button_cancel.clicked.connect(self.reject)
        self._button_ok = QPushButton('OK')
        self._button_ok.clicked.connect(self.accept)
        self._button_ok.setDefault(True)
        self._button_restore = QPushButton('Restore defaults')
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

    def keyPressEvent(self, e: 'QKeyEvent') -> None:
        if e.key() == Qt.Key.Key_Escape:
            # escape key should just close the window
            # which implies "accept"
            e.accept()
            self.reject()
            return
        super().keyPressEvent(e)

    def resizeEvent(self, event: 'QResizeEvent') -> None:
        """Override to emit signal."""
        self.resized.emit(event.size())
        super().resizeEvent(event)

    def _rebuild_dialog(self) -> None:
        """Removes settings not to be exposed to user and creates dialog pages."""

        self._starting_values = self._settings.model_dump(
            exclude={'schema_version'}
        )

        self._list.clear()  # Why recreate the list now? Why create it in the init?
        while self._stack.count():
            self._stack.removeWidget(self._stack.currentWidget())

        for (
            field_name,
            field_info,
        ) in self._settings.__class__.model_fields.items():
            field_type = get_inner_type(field_info.annotation)
            if get_origin(field_type) is None and issubclass(
                field_type, BaseModel
            ):
                self._add_page(field_name, field_info)
        item = QListWidgetItem('----Plugin Preferences----')
        item.setFlags(item.flags() & ~Qt.ItemIsSelectable & ~Qt.ItemIsEnabled)
        self._list.addItem(item)
        self._stack.addWidget(QLabel('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'))
        self.plugin_index = self._list.count()

        for plugin_name, plugin in self._plugin_settings.items():
            self._add_plugin(plugin_name, plugin)
        self._list.setCurrentRow(0)

    # def skip_plugin_pref(self):
    #     current = self._list.currentIndex().row()

    #     if self.plugin_index - 1 == current:
    #         next_index = current + 1
    #         if next_index < self._list.model().rowCount():
    #             self._list.setCurrentIndex(self._list.model().index(next_index, 0))

    def _add_plugin(
        self,
        plugin_name: str,
        plugin,
    ):
        """ "Builds the Plugin preferences widgets using the json schema builder.
        Similar to _add_page.
        This function only exists because it is possible to have multiple plugin configuration sets in one plugin.
        And they should all appear in the same place.

         Parameters
        ----------
        plugin_name : str
            the name of the plugin
        plugin : PluginPreferences
            the schemas containing multiple widgets for each plugin.
        """

        full = QWidget()
        layout = QVBoxLayout()
        name = plugin_name
        plugin_list = []
        for (
            field_name,
            field_info,
        ) in plugin.__class__.model_fields.items():
            field_type = get_inner_type(field_info.annotation)
            if get_origin(field_type) is None and issubclass(
                field_type, BaseModel
            ):
                schema, values = self._get_page_dict(
                    field_name, field_info, self._plugin_settings[plugin_name]
                )
                name = field_info.title or field_name
                form = self._widget_builder(
                    schema, values, name, self._plugin_settings[plugin_name]
                )
                plugin_list.append(form)
        full.setLayout(layout)
        [full.layout().addWidget(pl) for pl in plugin_list]
        page_scrollarea = QScrollArea()
        page_scrollarea.setWidgetResizable(True)
        page_scrollarea.setWidget(full)

        self._list.addItem(plugin.display_name)
        self._stack.addWidget(page_scrollarea)

    def _add_page(self, field_name: str, field_info: FieldInfo) -> None:
        """Builds the preferences widget using the json schema builder.

        Parameters
        ----------
        field_name : str
            the name of the plugin
        field_info : FieldInfo
            the schema to create the widget.
        """

        schema, values = self._get_page_dict(
            field_name, field_info, self._settings
        )
        name = field_info.title or field_name

        form = self._widget_builder(schema, values, name, self._settings)

        page_scrollarea = QScrollArea()
        page_scrollarea.setWidgetResizable(True)
        page_scrollarea.setWidget(form)

        self._list.addItem(name)
        self._stack.addWidget(page_scrollarea)

    def _widget_builder(self, schema, values, name, schema_object):
        """Creates a widget using a widget based on the schema."""
        from napari._vendor.qt_json_builder.qt_jsonschema_form import (
            WidgetBuilder,
        )

        form = WidgetBuilder().create_form(schema, self.ui_schema)
        # set state values for widget
        form.widget.state = values
        # make settings follow state of the form widget
        form.widget.on_changed.connect(
            lambda d: getattr(schema_object, name.lower()).update(d)
        )
        # make widgets follow values of the settings
        settings_category = getattr(schema_object, name.lower())
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
        return form

    def _get_page_dict(
        self, field_name: str, field_info: FieldInfo, settings_object: dict
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
        setting = getattr(settings_object, field_name)
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
            'Restore Settings',
            'Are you sure you want to restore default settings?',
            QMessageBox.StandardButton.RestoreDefaults
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.RestoreDefaults,
        )
        QApplication.instance().setAttribute(
            Qt.ApplicationAttribute.AA_DontUseNativeDialogs, prev
        )
        if response == QMessageBox.RestoreDefaults:
            self._settings.reset()
            for plugin_setting in self._plugin_settings.values():
                plugin_setting.reset()

    def _restart_required_dialog(self):
        """Displays the dialog informing user a restart is required."""
        QMessageBox.information(
            self,
            'Restart required',
            'A restart is required for some new settings to have an effect.',
        )

    def closeEvent(self, event: 'QCloseEvent') -> None:
        event.accept()
        self.reject()

    def accept(self):
        self._settings.save()
        for plugin_setting in self._plugin_settings.values():
            plugin_setting.save()
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
