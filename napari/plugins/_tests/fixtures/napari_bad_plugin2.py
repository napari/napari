"""
This plugin registers fine, and passes hook_specification validation.
But errors when the actual reader function is called.  This is used for testing
the read_data_with_plugins loop.

"""
import pluggy

napari_hook_implementation = pluggy.HookimplMarker("napari")


def reader_function(path):
    raise IOError(f"Plugin failed to read path: {path}")


@napari_hook_implementation
def napari_get_reader(path):
    if path.endswith('ext'):
        return reader_function
