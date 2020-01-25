"""Internal hooks to be registered by the plugin manager
"""
import pluggy


hookimpl = pluggy.HookimplMarker("napari")
"""Marker to be imported and used in plugins (and for own implementations)"""


@hookimpl(trylast=True)
def napari_get_reader(path: str):
    return lambda path: [(None, {'path': path})]
