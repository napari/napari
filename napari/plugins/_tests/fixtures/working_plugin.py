"""
Example plugin for testing plugin discovery and loading
"""
import pluggy

napari_hook_implementation = pluggy.HookimplMarker("napari")


def reader_function():
    pass


@napari_hook_implementation
def napari_get_reader(path):
    if path.endswith('true'):
        return reader_function
