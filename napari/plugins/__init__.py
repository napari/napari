from pathlib import Path

from npe2 import PluginManager as _PluginManager
from npe2 import PluginManifest
from npe2.manifest.package_metadata import PackageMetadata

from ..settings import get_settings

__all__ = ["menu_item_template"]


def _init_plugin_manager():
    _npe2pm = _PluginManager.instance()
    _npe2pm.discover()
    # this is a workaround for the fact that briefcase does not seem to
    # include napari's entry_points.txt in the bundled app, so the builtin plugins
    # don't get detected.  So we just register it manually.  This could potentially
    # be removed when we move to a different bundle strategy
    if 'napari' not in _npe2pm._manifests:
        mf_file = Path(__file__).parent.parent / 'builtins.yaml'
        mf = PluginManifest.from_file(mf_file)
        mf.package_metadata = PackageMetadata.for_package('napari')
        _npe2pm.register(mf)


_init_plugin_manager()

#: Template to use for namespacing a plugin item in the menu bar
menu_item_template = '{}: {}'
