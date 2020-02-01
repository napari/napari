import pluggy
from .manager import NapariPluginManager

hookimpl = pluggy.HookimplMarker("napari")
"""Marker to be imported and used in plugins (and for own implementations)
Note: plugins may also just import pluggy directly and make their own
napari hookimpl.
"""

# a singleton... but doesn't need to be.
# Could have seperate plugin managers for different interfaces if desired.
plugin_manager = NapariPluginManager()
