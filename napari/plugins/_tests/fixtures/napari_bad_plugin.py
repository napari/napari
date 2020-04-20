"""
Test plugin that is technically correct, but would cause performance problems
"""
from napari_plugins import napari_hook_implementation
import time


def reader_function(path):
    return True


@napari_hook_implementation
def napari_get_reader(path):
    time.sleep(1)  # this is too long!!
    if path.endswith('ext'):
        return reader_function
