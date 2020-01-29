import numpy as np
import pytest
from napari.utils.misc import callsignature
from napari._tests.utils import layer_test_data


@pytest.mark.parametrize('Layer, data, ndim', layer_test_data)
def test_attrs_arrays(Layer, data, ndim):
    """Test layer attributes and arrays."""
    np.random.seed(0)
    layer = Layer(data)
    # Check layer has been correctly created
    assert layer.ndim == ndim

    properties = layer._get_state()

    # Check every property is in call signature
    signature = callsignature(Layer)
    for prop in properties.keys():
        assert prop in signature.parameters

    # Check number of properties is same as number in signature
    assert len(properties) == len(signature.parameters)

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
