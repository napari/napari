"""
Test plugin that would fail to load.
"""
from napari_plugins import napari_hook_implementation


def reader_function(path):
    return True


@napari_hook_implementation
def napari_get_reader(path, arg1, arg2, i_just_love_args):
    # this has too many arguments!

    if path.endswith('ext'):
        return reader_function
