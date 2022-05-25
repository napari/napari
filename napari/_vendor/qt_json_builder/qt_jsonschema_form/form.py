from copy import deepcopy

from jsonschema.validators import validator_for

from . import widgets
from .defaults import compute_defaults


def get_widget_state(schema, state=None):
    if state is None:
        return compute_defaults(schema)
    return state


def get_schema_type(schema: dict) -> str:
    try:
        return schema['type']
    except KeyError:
        return 'object'


class WidgetBuilder:
    default_widget_map = {
        "boolean": {
            "checkbox": widgets.CheckboxSchemaWidget,
            "enum": widgets.EnumSchemaWidget,
        },
        "object": {
            "object": widgets.ObjectSchemaWidget,
            "enum": widgets.EnumSchemaWidget,
            "plugins": widgets.PluginWidget,
            "shortcuts": widgets.ShortcutsWidget,
            "extension2reader": widgets.Extension2ReaderWidget,
            "dask": widgets.DaskSettingsWidget,
        },
        "number": {
            "spin": widgets.SpinDoubleSchemaWidget,
            "text": widgets.TextSchemaWidget,
            "enum": widgets.EnumSchemaWidget,
        },
        "string": {
            "textarea": widgets.TextAreaSchemaWidget,
            "text": widgets.TextSchemaWidget,
            "password": widgets.PasswordWidget,
            "filepath": widgets.FilepathSchemaWidget,
            "colour": widgets.ColorSchemaWidget,
            "enum": widgets.EnumSchemaWidget,
        },
        "integer": {
            "spin": widgets.SpinSchemaWidget,
            "text": widgets.TextSchemaWidget,
            "range": widgets.IntegerRangeSchemaWidget,
            "enum": widgets.EnumSchemaWidget,
            "highlight": widgets.HighlightSizePreviewWidget,
        },
        "array": {
            "array": widgets.ArraySchemaWidget,
            "enum": widgets.EnumSchemaWidget,
        },
    }

    default_widget_variants = {
        "boolean": "checkbox",
        "object": "object",
        "array": "array",
        "number": "spin",
        "integer": "spin",
        "string": "text",
    }

    widget_variant_modifiers = {
        "string": lambda schema: schema.get("format", "text")
    }

    def __init__(self, validator_cls=None):
        self.widget_map = deepcopy(self.default_widget_map)
        self.validator_cls = validator_cls

    def create_form(
        self, schema: dict, ui_schema: dict, state=None
    ) -> widgets.SchemaWidgetMixin:

        validator_cls = self.validator_cls
        if validator_cls is None:
            validator_cls = validator_for(schema)

        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        schema_widget = self.create_widget(schema, ui_schema, state)

        form = widgets.FormWidget(schema_widget)

        def validate(data):
            form.clear_errors()

            errors = [*validator.iter_errors(data)]

            if errors:
                form.display_errors(errors)

            for err in errors:
                schema_widget.handle_error(err.path, err)

        schema_widget.on_changed.connect(validate)

        return form

    def create_widget(
        self,
        schema: dict,
        ui_schema: dict,
        state=None,
        description: str = "",
    ) -> widgets.SchemaWidgetMixin:
        schema_type = get_schema_type(schema)
        try:
            default_variant = self.widget_variant_modifiers[schema_type](
                schema
            )
        except KeyError:
            default_variant = self.default_widget_variants[schema_type]


        if "enum" in schema:
            default_variant = "enum"

        widget_variant = ui_schema.get('ui:widget', default_variant)
        widget_cls = self.widget_map[schema_type][widget_variant]
        widget = widget_cls(schema, ui_schema, self)
        default_state = get_widget_state(schema, state)
        if default_state is not None:
            widget.state = default_state

        if description:
            widget.setDescription(description)
            widget.setToolTip(description)

        return widget
