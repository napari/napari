import numpy as np
import pytest
from napari.utils.misc import callsignature
from napari.tests.utils import layer_test_data


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


@pytest.mark.parametrize('Layer, data, ndim', layer_test_data)
def test_layer_hash(Layer, data, ndim):
    """Test that layers can be compared.

    Layers with the same properties evaluate as equal (even if they are
    different objects).
    """
    layer1 = Layer(data)
    layer2 = Layer(data)
    assert id(layer1) != id(layer2)
    assert layer1 == layer2
    layer2.name = 'something else'
    assert layer1 != layer2
    assert layer2 == layer2
    layer1.name = 'something else'
    assert layer1 == layer2


image_test_data = [x for x in layer_test_data if x[0].__name__ == 'Image']


@pytest.mark.parametrize('Layer, data, ndim', image_test_data)
def test_layer_data_hash(Layer, data, ndim):
    """Test that data is included in the hash."""
    layer1 = Layer(data)
    layer2 = Layer(data)
    assert layer1 == layer2
    if isinstance(layer1.data, list):
        newdata = [np.random.random(arr.shape) for arr in layer1.data]
    else:
        newdata = np.random.random(layer1.data.shape)
    layer1.data = newdata
    assert layer1 != layer2
    layer2.data = newdata
    assert layer1 == layer2
