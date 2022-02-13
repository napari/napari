from functools import lru_cache
from pathlib import Path

from npe2 import PluginManager as _PluginManager
from npe2 import PluginManifest
from npe2.manifest.package_metadata import PackageMetadata

from ..settings import get_settings
from . import _npe2
from ._plugin_manager import NapariPluginManager

__all__ = ["plugin_manager", "menu_item_template"]

#: Template to use for namespacing a plugin item in the menu bar
menu_item_template = '{}: {}'
# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = NapariPluginManager()


@lru_cache()  # only call once
def _initialize_plugins():
    _npe2pm = _PluginManager.instance()
    _npe2pm.discover()
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

    plugin_manager._initialize()

    # Disable plugins listed as disabled in settings, or detected in npe2
    _from_npe2 = {m.package_metadata.name for m in _npe2.iter_manifests()}
    _toblock = get_settings().plugins.disabled_plugins.union(_from_npe2)
    plugin_manager._blocked.update(_toblock)


_initialize_plugins()
