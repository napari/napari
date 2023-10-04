from napari.layers import (
    Image,
    Layer
)

from napari._fastplotlib import (
    FastplotlibImageLayer
)

from napari.utils.translations import trans

layer_to_visual = {
    Image: FastplotlibImageLayer,
}


def create_fpl_layer(layer: Layer):
    """Create fpl visual for a layer based on its layer type.

    Parameters
    ----------
    layer : napari.layers._base_layer.Layer
        Layer that needs its property widget created.

    """
    for layer_type, visual_class in layer_to_visual.items():
        if isinstance(layer, layer_type):
            return visual_class(layer)

    raise TypeError(
        trans._(
            'Could not find FastplotlibLayer for layer of type {dtype}',
            deferred=True,
            dtype=type(layer),
        )
    )
