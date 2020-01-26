import pluggy
import pkgutil
from . import hookspecs, _builtins


hookimpl = pluggy.HookimplMarker("napari")
"""Marker to be imported and used in plugins (and for own implementations)"""

PLUGIN_PREFIX = "napari_"
PLUGIN_ENTRYPOINT = "napari.plugin"


def get_plugin_manager():
    # instantiate the plugin manager
    pm = pluggy.PluginManager("napari")
    # define hook specifications
    pm.add_hookspecs(hookspecs)
    # register modules defining the napari entry_point in setup.py
    pm.load_setuptools_entrypoints(PLUGIN_ENTRYPOINT)
    # register modules using naming convention
    for finder, name, ispkg in pkgutil.iter_modules():
        if name.startswith(PLUGIN_PREFIX):
            pm.register(importlib.import_module(name))
    # register our own built plugins
    pm.register(_builtins)
    return pm


# a singleton... but doesn't need to be.  Could have seperate plugin managers
# for different interfaces if desired.
plugin_manager = get_plugin_manager()
