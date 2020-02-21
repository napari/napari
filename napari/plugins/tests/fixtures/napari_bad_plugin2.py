"""
Test plugin that is technically correct, but would cause performance problems
"""
import pluggy

napari_hook_implementation = pluggy.HookimplMarker("napari")


def reader_function(path):
    raise IOError("whoops")


@napari_hook_implementation
def napari_get_reader(path):
    if path.endswith('ext'):
        return reader_function
