import importlib
import os
import pkgutil
from logging import Logger

import pluggy

from . import _builtins, hookspecs

logger = Logger(__name__)


class NapariPluginManager(pluggy.PluginManager):
    def __init__(self, autodiscover=True):
        """pluggy.PluginManager with napari-specific functionality

        In addition to the pluggy functionality, this subclass adds
        autodiscovery using package naming convention.

        Parameters
        ----------
        autodiscover : bool, optional
            Whether to autodiscover plugins by naming convention and setuptools
            entry_points, by default True
        """
        super().__init__("napari")

        # define hook specifications and validators
        self.add_hookspecs(hookspecs)

        # register our own built plugins
        self.register(_builtins, name='builtins')
        # discover external plugins
        if (
            not os.environ.get("NAPARI_DISABLE_PLUGIN_AUTOLOAD")
            and autodiscover
        ):
            self.discover()

    def discover(self):
        # avoid circular import
        from . import PLUGIN_ENTRYPOINT, PLUGIN_PREFIX

        count = 0
        if not os.environ.get("NAPARI_DISABLE_ENTRYPOINT_PLUGINS"):
            # register modules defining the napari entry_point in setup.py
            count += self.load_setuptools_entrypoints(PLUGIN_ENTRYPOINT)
        if not os.environ.get("NAPARI_DISABLE_NAMEPREFIX_PLUGINS"):
            # register modules using naming convention
            count += self.load_modules_by_prefix(PLUGIN_PREFIX)

        if count:
            msg = f'loaded {count} plugins:\n  '
            msg += "\n  ".join([n for n, m in self.list_name_plugin()])
            logger.info(msg)
        return count

    def load_modules_by_prefix(self, prefix):
        count = 0
        for finder, name, ispkg in pkgutil.iter_modules():
            if (
                not name.startswith(prefix)
                or self.get_plugin(name)
                or self.is_blocked(name)
            ):
                continue
            try:
                mod = importlib.import_module(name)
                self.register(mod, name=name)
                count += 1
            except Exception as e:
                logger.error(f'failed to import plugin: {name}: {str(e)}')
                self.unregister(mod)
        return count


NapariPluginManager.PluginValidationError = (
    pluggy.manager.PluginValidationError
)
