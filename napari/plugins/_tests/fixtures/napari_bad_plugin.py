"""
Test plugin that is technically correct, but would cause performance problems
"""
import pluggy
import time

napari_hook_implementation = pluggy.HookimplMarker("napari")


def reader_function(path):
    return True


@napari_hook_implementation
def napari_get_reader(path):
    time.sleep(1)  # this is too long!!
    if path.endswith('ext'):
        return reader_function
