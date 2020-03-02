from pluggy import HookimplMarker

from .manager import NapariPluginManager

# Marker to be imported and used in plugins (and for own implementations)
# Note: plugins may also just import pluggy directly and make their own
# napari_hook_implementation.
napari_hook_implementation = HookimplMarker("napari")

# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = NapariPluginManager()
