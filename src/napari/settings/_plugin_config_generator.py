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
    from npe2.manifest.contributions import ConfigurationContribution


def _field_name(key: str, plugin_name: str) -> str:
    """
    Convert:
        my_plugin.someSetting -> some_setting
    """
    key = key.removeprefix(f'{plugin_name}.')

    key = re.sub(
        r'(?<!^)(?=[A-Z])',
        '_',
        key,
    )

    key = re.sub(
        r'[.\-\s]+',
        '_',
        key,
    )

    return key.lower()


VALUE_TRANSLATOR = {
    'maximum': 'le',
    'minimum': 'ge',
    'exclusive_maximum': 'lt',
    'exclusive_minimum': 'gt',
}
_TYPE_MAP: dict[str, type] = {
    'boolean': bool,
    'string': str,
    'integer': int,
    'number': float,
    'array': list,
    'int': int,
    'float': float,
    'str': str,
    'bool': bool,
    'list': list,
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

        field_name = _field_name(key, plugin_name)

        fields[field_name] = (
            field_type,
            Field(**field_kwargs),
        )
    model_name = configuration.title.lower()
    model = create_model(
        model_name,
        __base__=EventedModel,
        **fields,
    )
    model.display = configuration.title  # pyright: ignore[reportAttributeAccessIssue]
    return model


def plugin_configuration_generator() -> dict[str, type[PluginPreferences]]:
    from npe2 import PluginManager

    pm = PluginManager.instance()
    pm.discover()
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
    plugin_settings = {}
    for plugin_name, configuration in configurations.items():
        models = [
            _build_single_config_model(conf, plugin_name)
            for conf in configuration
        ]
        fields = {}

        for model in models:
            fields[model.__name__] = (
                model,
                Field(default_factory=model, title=model.display),  # pyright: ignore[reportAttributeAccessIssue]
            )
        plugin_settings[plugin_name] = create_model(
            f'{plugin_name} Preferences',
            __base__=PluginPreferences,
            **fields,
        )
        plugin_settings[plugin_name].display_name = display_names[plugin_name]
    return plugin_settings
