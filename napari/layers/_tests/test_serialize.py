import inspect

import numpy as np
import pytest

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
    signature = inspect.signature(Layer)

    for prop in properties.keys():
        assert prop in signature.parameters

    # Check number of properties is same as number in signature
    # excluding affine transform which is not yet in `_get_state`
    assert len(properties) == len(signature.parameters) - 1

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
        elif isinstance(getattr(layer, prop), dict):
            assert np.all(
                [
                    np.all(value == getattr(new_layer, prop)[key])
                    for key, value in getattr(layer, prop).items()
                ]
            )
        else:
            assert np.all(getattr(layer, prop) == getattr(new_layer, prop))


@pytest.mark.parametrize('Layer, data, ndim', layer_test_data)
def test_no_callbacks(Layer, data, ndim):
    """Test no internal callbacks for layer emmitters."""
    layer = Layer(data)
    # Check layer has been correctly created
    assert layer.ndim == ndim

    # Check that no internal callbacks have been registered
    len(layer.events.callbacks) == 0
    for em in layer.events.emitters.values():
        assert len(em.callbacks) == 0
