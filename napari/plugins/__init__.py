from functools import lru_cache
from pathlib import Path

from npe2 import PackageMetadata
from npe2 import PluginManager as _PluginManager
from npe2 import PluginManifest

from ..settings import get_settings
from . import _npe2

__all__ = ["plugin_manager", "menu_item_template"]

#: Template to use for namespacing a plugin item in the menu bar
menu_item_template = '{}: {}'
# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = None


@lru_cache  # only call once
def _initialize_plugins():
    global plugin_manager

    _npe2pm = _PluginManager.instance()

    settings = get_settings()
    if settings.schema_version >= '0.4.0':
        for p in settings.plugins.disabled_plugins:
            _npe2pm.disable(p)

    _npe2pm.discover(include_npe1=settings.plugins.use_npe2_adaptor)
    _npe2pm.events.enablement_changed.connect(
        _npe2._on_plugin_enablement_change
    )

    # this is a workaround for the fact that briefcase does not seem to include
    # napari's entry_points.txt in the bundled app, so the builtin plugins
    # don't get detected.  So we just register it manually.  This could
    # potentially be removed when we move to a different bundle strategy
    if 'napari' not in _npe2pm._manifests:
        mf_file = Path(__file__).parent.parent / 'builtins.yaml'
        mf = PluginManifest.from_file(mf_file)
        mf.package_metadata = PackageMetadata.for_package('napari')
        _npe2pm.register(mf)

    # if we're using the npe2 "shim" adaptor, we don't use npe1 at all
    # we just make the plugin_manager a do-nothing mock object.
    # In a later version, we can make the npe2 adapter always-on and remove
    # the npe1 plugin_manager altogether.
    if settings.plugins.use_npe2_adaptor:
        from unittest.mock import MagicMock

        plugin_manager = MagicMock()
        return

    from ._plugin_manager import NapariPluginManager

    plugin_manager = NapariPluginManager()
    # Disable plugins listed as disabled in settings, or detected in npe2
    _from_npe2 = {m.name for m in _npe2pm.iter_manifests()}
    _from_npe2.add('napari')
    plugin_manager._skip_packages = _from_npe2

    plugin_manager._blocked.update(settings.plugins.disabled_plugins)
    plugin_manager._initialize()


_initialize_plugins()
