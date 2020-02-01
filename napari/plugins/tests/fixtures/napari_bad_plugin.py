"""
Test plugin that is technically correct, but would cause performance problems
"""
import pluggy
import time

hookimpl = pluggy.HookimplMarker("napari")


def reader_function():
    raise IOError('it worked')


@hookimpl
def napari_get_reader(path):
    time.sleep(1)  # this is too long!!
    if path.endswith('true'):
        return reader_function
