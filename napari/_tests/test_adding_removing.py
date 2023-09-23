import numpy as np
import pytest

from napari._tests.utils import (
    layer_test_data,
    skip_local_popups,
    skip_on_win_ci,
)
from napari.layers import Image
from napari.utils.events.event import WarningEmitter


@skip_on_win_ci
@skip_local_popups
@pytest.mark.parametrize('Layer, data, _', layer_test_data)
def test_add_all_layers(make_napari_viewer, Layer, data, _):
    """Make sure that all layers can show in the viewer."""
    viewer = make_napari_viewer(show=True)
    viewer.layers.append(Layer(data))


def test_layers_removed_on_close(make_napari_viewer):
    """Test layers removed on close."""
    viewer = make_napari_viewer()

    # add layers
    viewer.add_image(np.random.random((30, 40)))
    viewer.add_image(np.random.random((50, 20)))
    assert len(viewer.layers) == 2

    viewer.close()
    # check layers have been removed
    assert len(viewer.layers) == 0


def test_layer_multiple_viewers(make_napari_viewer):
    """Test layer on multiple viewers."""
    # Check that a layer can be added and removed from
    # mutliple viewers. See https://github.com/napari/napari/issues/1503
    # for more detail.
    viewer_a = make_napari_viewer()
    viewer_b = make_napari_viewer()

    # create layer
    layer = Image(np.random.random((30, 40)))
    # add layer
    viewer_a.layers.append(layer)
    viewer_b.layers.append(layer)

    # Change property
    layer.opacity = 0.8
    assert layer.opacity == 0.8

    # Remove layer from one viewer
    viewer_b.layers.remove(layer)

    # Change property
    layer.opacity = 0.6
    assert layer.opacity == 0.6


def test_adding_removing_layer(make_napari_viewer):
    """Test adding and removing a layer."""
    np.random.seed(0)
    viewer = make_napari_viewer()

    # Create layer
    data = np.random.random((2, 6, 30, 40))
    layer = Image(data)

    # Check that no internal callbacks have been registered
    assert len(layer.events.callbacks) == 0
    for em in layer.events.emitters.values():
        assert len(em.callbacks) == 0

    # Add layer
    viewer.layers.append(layer)
    np.testing.assert_array_equal(viewer.layers[0].data, data)
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 4
    # check that adding a layer created new callbacks
    assert any(len(em.callbacks) > 0 for em in layer.events.emitters.values())

    # Remove layer, viewer resets
    layer = viewer.layers[0]
    viewer.layers.remove(layer)
    assert len(viewer.layers) == 0
    assert viewer.dims.ndim == 2

    # Check that no other internal callbacks have been registered
    assert len(layer.events.callbacks) == 0
    for em in layer.events.emitters.values():
        assert len(em.callbacks) == 0

    # re-add layer
    viewer.layers.append(layer)
    assert len(viewer.layers) == 1
    assert viewer.dims.ndim == 4


@pytest.mark.parametrize('Layer, data, ndim', layer_test_data)
def test_add_remove_layer_external_callbacks(
    make_napari_viewer, Layer, data, ndim
):
    """Test external callbacks for layer emmitters preserved."""
    viewer = make_napari_viewer()

    layer = Layer(data)
    # Check layer has been correctly created
    assert layer.ndim == ndim

    # Connect a custom callback
    def my_custom_callback():
        return

    layer.events.connect(my_custom_callback)

    # Check that no internal callbacks have been registered
    assert len(layer.events.callbacks) == 1
    for em in layer.events.emitters.values():
        # warningEmitters are not connected when connecting to the emitterGroup
        if not isinstance(em, WarningEmitter):
            assert len(em.callbacks) == 1

    viewer.layers.append(layer)
    # Check layer added correctly
    assert len(viewer.layers) == 1
    # check that adding a layer created new callbacks
    assert any(len(em.callbacks) > 0 for em in layer.events.emitters.values())

    viewer.layers.remove(layer)
    # Check layer removed correctly
    assert len(viewer.layers) == 0

    # Check that all internal callbacks have been removed
    assert len(layer.events.callbacks) == 1
    for em in layer.events.emitters.values():
        # warningEmitters are not connected when connecting to the emitterGroup
        if not isinstance(em, WarningEmitter):
            assert len(em.callbacks) == 1
