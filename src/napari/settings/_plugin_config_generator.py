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


def _snake_identifier(name: str, plugin_name: str | None = None) -> str:
    """
    Convert free-form text into a valid snake_case Python identifier.

    Used to derive pydantic field/model names from npe2 configuration keys and
    titles, which are free-form text that must become valid Python identifiers
    (they are used both as the name of the generated model and as field names
    on the plugin preferences model).

    If ``plugin_name`` is given, a leading ``<plugin_name>.`` namespace is
    stripped from the key.  Plugin names and configuration keys may use
    different separators (e.g. ``'my-plugin'`` vs ``'my_plugin.reader.lazy'``),
    so the prefix is compared after normalizing separators.

    Examples
    --------
    >>> _snake_identifier('my_plugin.someSetting', 'my_plugin')
    'some_setting'
    >>> _snake_identifier('my_plugin.reader.lazy', 'my-plugin')
    'reader_lazy'
    >>> _snake_identifier('Demo Configuration for widget 1')
    'demo_configuration_for_widget_1'
    """
    if plugin_name:
        # configuration keys are commonly namespaced with the plugin name;
        # compare the two after splitting on non-alphanumeric runs so that
        # '-' and '_' separators are treated the same
        parts = re.split(r'[^0-9a-zA-Z]+', name)
        plugin_parts = re.split(r'[^0-9a-zA-Z]+', plugin_name)
        if parts[: len(plugin_parts)] == plugin_parts:
            name = '.'.join(parts[len(plugin_parts) :])

    # camelCase -> snake_case (e.g. someSetting -> some_Setting)
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
    # any run of non-alphanumeric characters is a separator
    name = re.sub(r'[^0-9a-zA-Z]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_').lower()
    if not name:
        name = 'settings'
    if name[0].isdigit():
        name = f'_{name}'
    return name


VALUE_TRANSLATOR = {
    'maximum': 'le',
    'minimum': 'ge',
    'exclusive_maximum': 'lt',
    'exclusive_minimum': 'gt',
}
# npe2 coerces configuration property types to JSON Schema type names
# (see ``npe2.manifest.contributions._json_schema``), so only those four
# need mapping
_TYPE_MAP: dict[str, type] = {
    'boolean': bool,
    'integer': int,
    'number': float,
    'string': str,
}


def _build_single_config_model(
    configuration: ConfigurationContribution,
    plugin_name: str,
) -> type[EventedModel]:

    fields: dict[str, Any] = {}

    for key, props in configuration.properties.items():
        if props.type is None:
            continue

        data = {k: getattr(props, k) for k in props.model_fields_set}

        type_name = data.pop('type')
        field_type = _TYPE_MAP.get(type_name)

        field_kwargs = {VALUE_TRANSLATOR.get(k, k): v for k, v in data.items()}

        field_name = _snake_identifier(key, plugin_name)

        fields[field_name] = (
            field_type,
            Field(**field_kwargs),
        )
    model = create_model(
        _snake_identifier(configuration.title),
        __base__=EventedModel,
        **fields,
    )
    model.display = configuration.title  # type: ignore[attr-defined]
    return model


def plugin_configuration_generator(
    plugin_manager: PluginManager | None = None,
) -> dict[str, type[PluginPreferences]]:
    if not plugin_manager:
        from npe2 import PluginManager

        pm = PluginManager.instance()
        pm.discover()
    else:
        pm = plugin_manager
    plugins = sorted(
        pm.iter_manifests(),
        key=lambda x: x.name,
    )
    display_names = {plugin.name: plugin.display_name for plugin in plugins}
    plugin_contr = {
        plug.name: plug.contributions for plug in plugins if plug.contributions
    }
    configurations = {
        plug: conf.configuration
        for plug, conf in plugin_contr.items()
        if conf.configuration
    }
    plugin_settings: dict[str, type[PluginPreferences]] = {}
    for plugin_name, configuration in configurations.items():
        models = [
            _build_single_config_model(conf, plugin_name)
            for conf in configuration
        ]
        fields: dict[str, Any] = {}

        for model in models:
            fields[model.__name__] = (
                model,
                Field(default_factory=model, title=model.display),  # type: ignore[attr-defined]
            )
        plugin_settings[plugin_name] = create_model(
            f'{plugin_name} Preferences',
            __base__=PluginPreferences,
            **fields,
        )
        plugin_settings[plugin_name].display_name = display_names[plugin_name]
    return plugin_settings
