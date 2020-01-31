import pluggy
from .manager import plugin_manager

hookimpl = pluggy.HookimplMarker("napari")
"""Marker to be imported and used in plugins (and for own implementations)"""
