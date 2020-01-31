import importlib
import os
import pkgutil
from logging import Logger

import pluggy

from . import _builtins, hookspecs

logger = Logger(__name__)

PLUGIN_ENTRYPOINT = "napari.plugin"


def find_module_by_prefix(prefix="napari_"):
    modules = []
    for finder, name, ispkg in pkgutil.iter_modules():
        if name.startswith(prefix):
            try:
                modules.append(importlib.import_module(name))
            except Exception as e:
                logger.error(f'failed to import plugin: {name}: {str(e)}')
    return modules


def get_plugin_manager():
    # instantiate the plugin manager
    pm = pluggy.PluginManager("napari")
    # define hook specifications
    pm.add_hookspecs(hookspecs)

    # register our own built plugins
    pm.register(_builtins)

    count = 0
    if not os.environ.get("NAPARI_DISABLE_PLUGIN_AUTOLOAD"):
        # register modules defining the napari entry_point in setup.py
        count += pm.load_setuptools_entrypoints(PLUGIN_ENTRYPOINT)

        # register modules defining the napari entry_point in setup.py
        for module in find_module_by_prefix():
            try:
                pm.register(module)
                count += 1
            except Exception as e:
                logger.error(
                    f'failed register plugin module: {module.__name__}: {e}'
                )
    if count:
        msg = f'loaded {count} plugins:\n  '
        msg += "\n  ".join([n for n, m in pm.list_name_plugin()])
        logger.info(msg)

    return pm


# a singleton... but doesn't need to be.  Could have seperate plugin managers
# for different interfaces if desired.
plugin_manager = get_plugin_manager()
