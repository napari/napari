"""
Example plugin for testing plugin discovery and loading
"""
from napari_plugins import napari_hook_implementation


def reader_function(path):
    return True


@napari_hook_implementation
def napari_get_reader(path):
    if path.endswith('ext'):
        return reader_function
