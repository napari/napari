import numpy as np

from napari.components import LayerList
from napari.layers import Image
from napari._qt.qt_layerlist import QtLayerList
from napari._tests.utils import assert_layerlist_model_view_synced


def test_creating_empty_view(qtbot):
    """Test creating LayerList view."""
    layers = LayerList()
    view = QtLayerList(layers)

    qtbot.addWidget(view)

    # Check that the layers model has been appended to the layers view
    assert view.layers == layers
    assert_layerlist_model_view_synced(view, layers)


def test_adding_layers(qtbot):
    """Test adding layers."""
    layers = LayerList()
    view = QtLayerList(layers)

    qtbot.addWidget(view)

    # Check that new layer and divider get added to vbox_layout
    layer_a = Image(np.random.random((10, 10)))
    layers.append(layer_a)
    assert_layerlist_model_view_synced(view, layers)

    # Check that new layers and dividers get added to vbox_layout
    layer_b = Image(np.random.random((15, 15)))
    layer_c = Image(np.random.random((15, 15)))
    layer_d = Image(np.random.random((15, 15)))
    layers.append(layer_b)
    layers.append(layer_c)
    layers.append(layer_d)
    assert_layerlist_model_view_synced(view, layers)
    assert view.layers == layers


def test_removing_layers(qtbot):
    """Test removing layers."""
    layers = LayerList()
    view = QtLayerList(layers)

    qtbot.addWidget(view)

    layer_a = Image(np.random.random((10, 10)))
    layer_b = Image(np.random.random((15, 15)))
    layer_c = Image(np.random.random((15, 15)))
    layer_d = Image(np.random.random((15, 15)))
    layers.append(layer_a)
    layers.append(layer_b)
    layers.append(layer_c)
    layers.append(layer_d)

    # Check layout and layers list match after removing a layer
    layers.remove(layer_b)
    assert_layerlist_model_view_synced(view, layers)

    # Check layout and layers list match after removing a layer
    layers.remove(layer_d)
    assert_layerlist_model_view_synced(view, layers)

    layers.append(layer_b)
    layers.append(layer_d)
    # Select first and third layers
    for l, s in zip(layers, [True, True, False, False]):
        l.selected = s
    layers.remove_selected()
    assert_layerlist_model_view_synced(view, layers)
    assert view.layers == layers


def test_reordering_layers(qtbot):
    """Test reordering layers."""
    layers = LayerList()
    view = QtLayerList(layers)

    qtbot.addWidget(view)

    layer_a = Image(np.random.random((10, 10)), name='image_a')
    layer_b = Image(np.random.random((15, 15)), name='image_b')
    layer_c = Image(np.random.random((15, 15)), name='image_c')
    layer_d = Image(np.random.random((15, 15)), name='image_d')
    layers.append(layer_a)
    layers.append(layer_b)
    layers.append(layer_c)
    layers.append(layer_d)

    # Check layout and layers list match after rearranging layers
    layers[:] = layers[(1, 0, 3, 2)]
    assert_layerlist_model_view_synced(view, layers)

    # Check layout and layers list match after swapping two layers
    layers['image_b', 'image_c'] = layers['image_c', 'image_b']
    assert_layerlist_model_view_synced(view, layers)

    # Check layout and layers list match after reversing list
    layers.reverse()
    assert_layerlist_model_view_synced(view, layers)
    assert view.layers == layers
