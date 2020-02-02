import pluggy
from .manager import NapariPluginManager

hookimpl = pluggy.HookimplMarker("napari")
"""Marker to be imported and used in plugins (and for own implementations)
Note: plugins may also just import pluggy directly and make their own
napari hookimpl.
"""


plugin_manager = NapariPluginManager()
