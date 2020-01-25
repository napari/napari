import pluggy
import pkgutil
from . import hookspecs, _internal


hookimpl = pluggy.HookimplMarker("napari")
"""Marker to be imported and used in plugins (and for own implementations)"""

PLUGIN_PREFIX = "napari_"
PLUGIN_ENTRYPOINT = "napari.plugin"


def get_plugin_manager():
    pm = pluggy.PluginManager("napari")
    pm.add_hookspecs(hookspecs)
    pm.load_setuptools_entrypoints(PLUGIN_ENTRYPOINT)
    for finder, name, ispkg in pkgutil.iter_modules():
        if name.startswith(PLUGIN_PREFIX):
            pm.register(importlib.import_module(name))
    pm.register(_internal)
    return pm


plugin_manager = get_plugin_manager()
