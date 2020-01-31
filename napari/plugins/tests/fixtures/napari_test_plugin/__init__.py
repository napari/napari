"""
Example plugin structure for testing plugin discovery
"""
import pluggy

hookimpl = pluggy.HookimplMarker("napari")


def reader_function():
    raise IOError('it worked')


@hookimpl()
def napari_get_reader(path, df):
    if str.endswith('true'):
        return reader_function
