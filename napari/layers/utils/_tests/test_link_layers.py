import numpy as np
import pytest

from napari import layers
from napari.layers.utils._link_layers import (
    layers_linked,
    link_layers,
    unlink_layers,
)

BASE_ATTRS = {}
BASE_ATTRS = {
    'opacity': 0.75,
    'blending': 'additive',
    'visible': False,
    'editable': False,
    'shear': [30],
}

IM_ATTRS = {
    'rendering': 'translucent',
    'iso_threshold': 0.34,
    'interpolation': 'bilinear',
    'contrast_limits': [0.25, 0.75],
    'gamma': 0.5,
}


@pytest.mark.parametrize('key, value', {**BASE_ATTRS, **IM_ATTRS}.items())
def test_link_image_layers_all_attributes(key, value):
    """Test linking common attributes across layers of similar types."""
    l1 = layers.Image(np.random.rand(10, 10), contrast_limits=(0, 0.8))
    l2 = layers.Image(np.random.rand(10, 10), contrast_limits=(0.1, 0.9))
    link_layers([l1, l2])
    # linking does (currently) apply to things that were unequal before linking
    assert l1.contrast_limits != l2.contrast_limits

    # once we set either... they will both be changed
    assert getattr(l1, key) != value
    setattr(l2, key, value)
    assert getattr(l1, key) == getattr(l2, key) == value


@pytest.mark.parametrize('key, value', BASE_ATTRS.items())
def test_link_different_type_layers_all_attributes(key, value):
    """Test linking common attributes across layers of different types."""
    l1 = layers.Image(np.random.rand(10, 10))
    l2 = layers.Points(None)
    link_layers([l1, l2])

    # once we set either... they will both be changed
    assert getattr(l1, key) != value
    setattr(l2, key, value)
    assert getattr(l1, key) == getattr(l2, key) == value


def test_link_invalid_param():
    """Test that linking non-shared attributes raises."""
    l1 = layers.Image(np.random.rand(10, 10))
    l2 = layers.Points(None)
    with pytest.raises(ValueError) as e:
        link_layers([l1, l2], ('rendering',))
    assert "Cannot link attributes that are not shared by all layers" in str(e)


def test_double_linking_noop():
    """Test that linking already linked layers is a noop."""
    l1 = layers.Points(None)
    l2 = layers.Points(None)
    l3 = layers.Points(None)
    # no callbacks to begin with
    assert len(l1.events.opacity.callbacks) == 0

    # should have two after linking layers
    link_layers([l1, l2, l3])
    assert len(l1.events.opacity.callbacks) == 2

    # should STILL have two after linking layers again
    link_layers([l1, l2, l3])
    assert len(l1.events.opacity.callbacks) == 2


def test_removed_linked_target():
    """Test that linking already linked layers is a noop."""
    l1 = layers.Points(None)
    l2 = layers.Points(None)
    l3 = layers.Points(None)
    link_layers([l1, l2, l3])

    l1.opacity = 0.5
    assert l1.opacity == l2.opacity == l3.opacity == 0.5

    # if we delete layer3 we shouldn't get an error when updating otherlayers
    del l3
    l1.opacity = 0.25
    assert l1.opacity == l2.opacity


def test_context_manager():
    """Test that we can temporarily link layers."""
    l1 = layers.Points(None)
    l2 = layers.Points(None)
    l3 = layers.Points(None)
    assert len(l1.events.opacity.callbacks) == 0
    with layers_linked([l1, l2, l3], ('opacity',)):
        assert len(l1.events.opacity.callbacks) == 2
        assert len(l1.events.blending.callbacks) == 0  # it's just opacity
        del l2  # if we lose a layer in the meantime it should be ok
    assert len(l1.events.opacity.callbacks) == 0


def test_unlink_layers():
    """Test that we can unlink layers."""
    l1 = layers.Points(None)
    l2 = layers.Points(None)
    l3 = layers.Points(None)

    link_layers([l1, l2, l3])
    assert len(l1.events.opacity.callbacks) == 2
    unlink_layers([l1, l2], ('opacity',))  # just unlink opacity on l1/l2

    assert len(l1.events.opacity.callbacks) == 1
    assert len(l2.events.opacity.callbacks) == 1
    # l3 is still connected to them both
    assert len(l3.events.opacity.callbacks) == 2
    # blending was untouched
    assert len(l1.events.blending.callbacks) == 2
    assert len(l2.events.blending.callbacks) == 2
    assert len(l3.events.blending.callbacks) == 2

    unlink_layers([l1, l2, l3])  # unlink everything
    assert len(l1.events.blending.callbacks) == 0
    assert len(l2.events.blending.callbacks) == 0
    assert len(l3.events.blending.callbacks) == 0


def test_unlink_single_layer():
    """Test that we can unlink a single layer from all others."""
    l1 = layers.Points(None)
    l2 = layers.Points(None)
    l3 = layers.Points(None)

    link_layers([l1, l2, l3])
    assert len(l1.events.opacity.callbacks) == 2
    unlink_layers([l1], ('opacity',))  # just unlink L1 opacicity from others
    assert len(l1.events.opacity.callbacks) == 0
    assert len(l2.events.opacity.callbacks) == 1
    assert len(l3.events.opacity.callbacks) == 1

    # blending was untouched
    assert len(l1.events.blending.callbacks) == 2
    assert len(l2.events.blending.callbacks) == 2
    assert len(l3.events.blending.callbacks) == 2

    unlink_layers([l1])  # completely unlink L1 from everything
    assert not l1.events.blending.callbacks
