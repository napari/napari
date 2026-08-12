from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import (
    Field,
    create_model,
)

from napari.settings._plugin_preferences import PluginPreferences
from napari.utils.events import EventedModel

if TYPE_CHECKING:
    from npe2 import PluginManager
    from npe2.manifest.contributions import ConfigurationContribution

# npe2 declares constraints using JSON Schema names; translate them to the
# equivalent pydantic ``Field`` kwargs.  All other keys (``title``, ``default``,
# ``description``, ``enum``, ...) are passed through unchanged and end up in
# ``json_schema_extra``.
# TODO: Check if this correctly is used by the widgets
VALUE_TRANSLATOR = {
    'maximum': 'le',
    'minimum': 'ge',
    'exclusive_maximum': 'lt',
    'exclusive_minimum': 'gt',
}


def _model_name(plugin_name: str) -> str:
    """Derive a valid Python identifier from a plugin name for the pydantic model.

    Manifest ``name`` values are validated by npe2 as PEP-508 package names
    (which may contain ``-`` and ``.``), not as Python identifiers — e.g.
    ``'my-plugin'`` is a legal plugin name.  The generated pydantic class name
    must still be a valid identifier (older pydantic versions fail on invalid
    model names), so the name is sanitized for the class name only.  This does
    NOT affect attribute/field names, which come verbatim from npe2-validated
    keys.
    """
    name = re.sub(r'\W+', '_', plugin_name).strip('_')
    if not name:
        name = 'plugin'
    if name[0].isdigit():
        name = f'_{name}'
    return name


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
        if props.type is None:
            continue

        data = {k: getattr(props, k) for k in props.model_fields_set}
        data.pop('type', None)

        field_kwargs = {VALUE_TRANSLATOR.get(k, k): v for k, v in data.items()}

        fields[key] = (
            props.python_type,
            Field(**field_kwargs),
        )
    return create_model(
        conf_identifier,
        __base__=EventedModel,
        **fields,
    )


def plugin_configuration_generator(
    plugin_manager: PluginManager | None = None,
) -> dict[str, type[PluginPreferences]]:
    """Build a plugin-preferences model class for each plugin with configurations."""
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
    plugin_settings: dict[str, type[PluginPreferences]] = {}
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
            __base__=PluginPreferences,
            **fields,
        )
        plugin_settings[plugin_name].display_name = display_names[  # type: ignore[attr-defined]
            plugin_name
        ]
    return plugin_settings
