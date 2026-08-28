from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Any, cast

from pydantic import (
    AfterValidator,
    Field,
    create_model,
)

from napari.settings._plugin_settings import PluginSettings
from napari.utils.events import EventedModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from npe2 import PluginManager
    from npe2.manifest.contributions import ConfigurationContribution

# npe2 declares constraints using JSON Schema names; translate them to the
# equivalent pydantic ``Field`` kwargs (``maximum`` -> ``le``, etc.).
VALUE_TRANSLATOR = {
    'maximum': 'le',
    'minimum': 'ge',
    'exclusive_maximum': 'lt',
    'exclusive_minimum': 'gt',
}

# The pydantic ``Field`` kwargs we may pass through, after ``VALUE_TRANSLATOR``.
# Anything else npe2 puts on a ``ConfigurationProperty`` (``enum``,
# ``is_multiline``, ``enum_descriptions``, ``schema_``, ...) is settings-UI
# metadata: it is not a valid ``Field`` kwarg, so it is routed to
# ``json_schema_extra`` instead (where it stays visible in the model's JSON
# schema for the Preferences widgets) rather than triggering pydantic's
# deprecation warning for extra keyword arguments.  ``enum`` additionally gets
# an ``AfterValidator`` (see ``_build_single_config_model``) so that out-of-enum
# values are rejected at runtime.
_FIELD_KWARGS = {
    'default',
    'title',
    'description',
    'gt',
    'ge',
    'lt',
    'le',
    'multiple_of',
    'min_length',
    'max_length',
}


def _model_name(plugin_name: str) -> str:
    """Derive a valid Python identifier from a plugin name for the pydantic model.

    Manifest ``name`` values are validated by npe2 as PEP-508 package names
    (which may contain ``-`` and ``.``, and may start with a digit), not as
    Python identifiers — e.g. ``'9lives'`` is a legal plugin name.  The
    generated pydantic class name must still be a valid identifier (older
    pydantic versions fail on invalid model names), so the name is sanitized
    for the class name only, prefixing ``_`` when it starts with a digit.  This
    does NOT affect attribute/field names, which come verbatim from
    npe2-validated keys.
    """
    name = re.sub(r'\W+', '_', plugin_name).strip('_')
    if name[0].isdigit():
        name = f'_{name}'
    return name


def _enum_validator(enum: list) -> Callable[[Any], Any]:
    """Return a validator that enforces membership in ``enum``."""

    def _validate(value: Any) -> Any:
        if value not in enum:
            raise ValueError(
                f'value {value!r} is not one of the allowed enum values '
                f'{enum!r}'
            )
        return value

    return _validate


def _build_single_config_model(
    configuration: ConfigurationContribution,
    conf_identifier: str,
) -> type[EventedModel]:
    """Build an :class:`EventedModel` for a single configuration category.

    npe2 guarantees that configuration and property keys are valid, non-reserved
    Python identifiers that don't start with an underscore (see
    ``npe2.manifest._validators.configuration_key``), and uses them verbatim as
    attribute names on the generated settings model, so no normalization is
    needed here.
    """

    fields: dict[str, Any] = {}

    for key, props in configuration.properties.items():
        data = {k: getattr(props, k) for k in props.model_fields_set}
        data.pop('type', None)

        # ``enum`` is both settings-UI metadata (renders a dropdown) and a
        # validation constraint. ``enum`` stays in ``data`` so it is routed to
        # ``json_schema_extra`` for the dropdown widget; membership in it is
        # enforced by an explicit ``AfterValidator`` (the generated models
        # inherit ``EventedModel``'s ``validate_assignment``, so out-of-enum
        # values are rejected on assignment).
        field_type: Any = props.python_type
        if data.get('enum'):
            field_type = Annotated[
                props.python_type,
                AfterValidator(_enum_validator(data['enum'])),
            ]

        field_kwargs = {
            VALUE_TRANSLATOR.get(k, k): v
            for k, v in data.items()
            if VALUE_TRANSLATOR.get(k, k) in _FIELD_KWARGS
        }
        extra = {
            k: v
            for k, v in data.items()
            if VALUE_TRANSLATOR.get(k, k) not in _FIELD_KWARGS
        }
        if extra:
            field_kwargs['json_schema_extra'] = {
                **field_kwargs.get('json_schema_extra', {}),
                **extra,
            }

        fields[key] = (
            field_type,
            Field(**field_kwargs),
        )
    return create_model(
        conf_identifier,
        __base__=EventedModel,
        **fields,
    )


def plugin_configuration_generator(
    plugin_manager: PluginManager | None = None,
) -> dict[str, type[PluginSettings]]:
    """Build a plugin-settings model class for each plugin with configurations."""
    if plugin_manager is None:
        from npe2 import PluginManager

        pm = PluginManager.instance()
        pm.discover()
    else:
        pm = plugin_manager
    # exclude disabled plugins, consistent with the rest of napari
    plugins = sorted(
        pm.iter_manifests(disabled=False),
        key=lambda x: x.name,
    )
    display_names = {plugin.name: plugin.display_name for plugin in plugins}
    configurations = {
        plug.name: plug.contributions.configurations
        for plug in plugins
        if plug.contributions.configurations
    }
    plugin_settings: dict[str, type[PluginSettings]] = {}
    for plugin_name, configuration in configurations.items():
        fields: dict[str, Any] = {}
        for conf_name, conf in configuration.items():
            model = _build_single_config_model(conf, conf_name)
            fields[conf_name] = (
                model,
                Field(default_factory=model, title=conf.title),
            )
        plugin_settings[plugin_name] = create_model(
            _model_name(plugin_name),
            __base__=PluginSettings,
            **fields,
        )
        # ``display_name`` is plugin metadata (not a settings field): it is set
        # on the dynamically-created model class so the Preferences dialog can
        # read it off each instance. ``cast`` is needed because mypy cannot
        # know about attributes added to a class generated at runtime.
        cast(Any, plugin_settings[plugin_name]).display_name = display_names[
            plugin_name
        ]
    return plugin_settings
