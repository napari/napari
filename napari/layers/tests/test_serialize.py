import numpy as np
import pytest
from napari import layers
from napari.util.misc import callsignature


np.random.seed(0)
surface_data = (
    np.random.random((10, 2)),
    np.random.randint(10, size=(6, 3)),
    np.random.random(10),
)
input = [
    (layers.Image, np.random.random((15, 30))),
    (layers.Labels, np.random.randint(20, size=(30, 30))),
    (layers.Points, 20 * np.random.random((10, 2))),
    (layers.Shapes, 20 * np.random.random((10, 4, 2))),
    (layers.Surface, surface_data),
    (layers.Vectors, np.random.random((10, 2, 2))),
]


@pytest.mark.parametrize('Layer,data', input)
def test_attrs_arrays(Layer, data):
    """Test layer attributes and arrays."""
    layer = Layer(data)
    properties = layer.attrs
    properties.update(layer.arrays)

    # Check layer_type present and correct
    assert 'layer_type' in properties
    assert properties['layer_type'] == Layer.__name__

    # Remove layer_type from properties
    del properties['layer_type']

    # Check every remaining property is in call signature
    signature = callsignature(Layer)
    for prop in properties.keys():
        assert prop in signature.parameters

    # Check new layer can be created
    new_layer = Layer(**properties)

    # Check that new layer matches old on all properties:
    for prop in properties.keys():
        # If lists check equality of all elements with np.all
        if isinstance(getattr(layer, prop), list):
            assert np.all(
                [
                    np.all(ol == nl)
                    for ol, nl in zip(
                        getattr(layer, prop), getattr(new_layer, prop)
                    )
                ]
            )
        else:
            assert np.all(getattr(layer, prop) == getattr(new_layer, prop))
