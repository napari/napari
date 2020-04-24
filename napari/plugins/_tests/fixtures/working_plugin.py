"""
Example plugin for testing plugin discovery and loading
"""
from napari_plugins import napari_hook_implementation


def reader_function():
    pass


@napari_hook_implementation
def napari_get_reader(path):
    if path.endswith('true'):
        return reader_function
