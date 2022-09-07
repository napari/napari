"""
Reader plugin
=============

Barebones reader plugin example, using ``imageio.imread```

.. tags:: historical
"""
from napari_plugin_engine import napari_hook_implementation
from imageio import formats, imread


readable_extensions = tuple({x for f in formats for x in f.extensions})


@napari_hook_implementation
def napari_get_reader(path):
    """A basic implementation of the napari_get_reader hook specification."""
    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(readable_extensions):
        return None
    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path and returns a list of LayerData tuples."""
    data = imread(path)
    # Readers are expected to return data as a list of tuples, where each tuple
    # is (data, [meta_dict, [layer_type]])
    return [(data,)]
