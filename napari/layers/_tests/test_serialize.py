import inspect

import numpy as np
import pytest

from napari._tests.utils import (
    are_objects_equal,
    count_warning_events,
    layer_test_data,
)
from napari.layers import Layer


@pytest.mark.parametrize(('LayerType', 'data', 'ndim'), layer_test_data)
def test_attrs_arrays(LayerType: type[Layer], data, ndim):
    """Test layer attributes and arrays."""
    np.random.seed(0)
    layer = LayerType(data)
    # Check layer has been correctly created
    assert layer.ndim == ndim

    properties = layer._get_state()

    # Check every property is in call signature
    signature = inspect.signature(LayerType)

    # Remove deprecated properties because that's not the main goal here.
    for deprecated in properties.deprecations:
        del properties[deprecated]
    signature = signature.replace(
        parameters=tuple(
            param
            for param in signature.parameters.values()
            if param.name not in properties.deprecations
        )
    )

    # Check every property is also a parameter.
    for prop in properties:
        assert prop in signature.parameters

    # Check number of properties is same as number in signature
    # excluding `cache` which is not yet in `_get_state`
    assert len(properties) == len(signature.parameters) - 1

    # Check new layer can be created
    new_layer = LayerType(**properties)

    # Check that new layer matches old on all properties:
    for prop in properties:
        assert are_objects_equal(
            getattr(layer, prop), getattr(new_layer, prop)
        )


@pytest.mark.parametrize(('LayerType', 'data', 'ndim'), layer_test_data)
def test_no_callbacks(LayerType: type[Layer], data, ndim):
    """Test no internal callbacks for layer emitters."""
    layer = LayerType(data)
    # Check layer has been correctly created
    assert layer.ndim == ndim

    # Check that no internal callbacks have been registered
    assert len(layer.events.callbacks) == 0
    for em in layer.events.emitters.values():
        assert len(em.callbacks) == count_warning_events(em.callbacks)
