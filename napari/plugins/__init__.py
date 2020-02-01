import pluggy
from .manager import NapariPluginManager

PLUGIN_ENTRYPOINT = "napari.plugin"
PLUGIN_PREFIX = "napari_"

hookimpl = pluggy.HookimplMarker("napari")
"""Marker to be imported and used in plugins (and for own implementations)"""

# a singleton... but doesn't need to be.  Could have seperate plugin managers
# for different interfaces if desired.
plugin_manager = NapariPluginManager()
