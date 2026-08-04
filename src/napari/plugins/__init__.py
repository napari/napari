from functools import lru_cache

from npe2 import (
    PluginManager as _PluginManager,
)

from napari.plugins import _npe2
from napari.plugins.environments import (
    PluginEnvironmentError,
    PluginEnvironmentInfo,
    PluginEnvironmentOperation,
    PluginEnvironmentOperationRecord,
    PluginEnvironmentProvisioningError,
    PluginEnvironmentState,
    PluginEnvironmentUnavailableError,
    PluginTask,
    PluginTaskCanceledError,
    PluginTaskMetadata,
    PluginTaskPhase,
    PluginTaskProgress,
    PluginTaskState,
    PluginWorkerError,
    PluginWorkerFailure,
    PluginWorkerState,
    add_plugin_environment_operation_callback,
    clear_plugin_environment_operations,
    execute_worker_command,
    list_active_plugin_environment_tasks,
    list_plugin_environment_operations,
    list_plugin_environments,
    prepare_plugin_environment,
    remove_plugin_environments,
    stop_plugin_workers,
)
from napari.settings import get_settings

__all__ = (
    'PluginEnvironmentError',
    'PluginEnvironmentInfo',
    'PluginEnvironmentOperation',
    'PluginEnvironmentOperationRecord',
    'PluginEnvironmentProvisioningError',
    'PluginEnvironmentState',
    'PluginEnvironmentUnavailableError',
    'PluginTask',
    'PluginTaskCanceledError',
    'PluginTaskMetadata',
    'PluginTaskPhase',
    'PluginTaskProgress',
    'PluginTaskState',
    'PluginWorkerError',
    'PluginWorkerFailure',
    'PluginWorkerState',
    'add_plugin_environment_operation_callback',
    'clear_plugin_environment_operations',
    'execute_worker_command',
    'list_active_plugin_environment_tasks',
    'list_plugin_environment_operations',
    'list_plugin_environments',
    'menu_item_template',
    'plugin_manager',
    'prepare_plugin_environment',
    'remove_plugin_environments',
    'stop_plugin_workers',
)

from napari.utils.theme import _install_npe2_themes

#: Template to use for namespacing a plugin item in the menu bar
# widget_name (plugin_name)
menu_item_template = '{1} ({0})'
"""Template to use for namespacing a plugin item in the menu bar"""


@lru_cache  # only call once
def _initialize_plugins() -> None:
    _npe2pm = _PluginManager.instance()

    settings = get_settings()
    if settings.schema_version >= '0.4.0':
        for p in settings.plugins.disabled_plugins:
            _npe2pm.disable(p)

    # just in case anything has already been registered before we initialized
    _npe2.on_plugins_registered(set(_npe2pm.iter_manifests()))

    # connect enablement/registration events to listeners
    _npe2pm.events.enablement_changed.connect(
        _npe2.on_plugin_enablement_change
    )
    _npe2pm.events.plugins_registered.connect(_npe2.on_plugins_registered)
    _npe2pm.discover(include_npe1=True)

    # Disable plugins listed as disabled in settings, or detected in npe2
    _from_npe2 = {m.name for m in _npe2pm.iter_manifests()}
    if 'napari' in _from_npe2:
        _from_npe2.update({'napari', 'builtins'})

    _install_npe2_themes()
