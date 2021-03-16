import numpy as np
from qtpy.QtCore import QModelIndex
from vispy.color import get_color_dict

from napari._qt.containers.layers import QtLayerList
from napari.components import LayerList
from napari.layers import Image
from napari.utils.colormaps.standardize_color import hex_to_name


def check_layout_layers(view: QtLayerList, layers):
    """
    Check the layer widget order matches the layers order in the layout

    Parameters
    ----------
    layout : QLayout
        Layout to test
    layers : napari.components.LayerList
        LayersList to compare to

    Returns
    -------
    match : bool
        Boolean if layout matches layers
    """
    model = view.model()
    model_layers = [
        model.getItem(model.index(i, 0, QModelIndex()))
        for i in range(model.rowCount())
    ]
    return model_layers == list(layers)


def test_creating_empty_view(qtbot):
    """
    Test creating LayerList view.
    """
    layers = LayerList()
    view = QtLayerList(layers)

    qtbot.addWidget(view)

    # Check that the layers model has been appended to the layers view
    assert view.layers == layers
    assert view.model().rowCount() == 0
    assert check_layout_layers(view, layers)


def test_adding_layers(qtbot):
    """
    Test adding layers.
    """
    layers = LayerList()
    view = QtLayerList(layers)

    qtbot.addWidget(view)

    # Check that new layer and divider get added to view
    layer_a = Image(np.random.random((10, 10)))
    layers.append(layer_a)
    assert view.model().rowCount() == len(layers)
    assert check_layout_layers(view, layers)

    # Check that new layers and dividers get added to view
    layer_b = Image(np.random.random((15, 15)))
    layer_c = Image(np.random.random((15, 15)))
    layer_d = Image(np.random.random((15, 15)))
    layers.append(layer_b)
    layers.append(layer_c)
    layers.append(layer_d)
    assert view.model().rowCount() == len(layers)
    assert check_layout_layers(view, layers)


def test_removing_layers(qtbot):
    """
    Test removing layers.
    """
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
    assert view.model().rowCount() == len(layers)
    assert check_layout_layers(view, layers)

    # Check layout and layers list match after removing a layer
    layers.remove(layer_d)
    assert view.model().rowCount() == len(layers)
    assert check_layout_layers(view, layers)

    layers.append(layer_b)
    layers.append(layer_d)
    # Select first and third layers
    for layer, s in zip(layers, [True, True, False, False]):
        layers.selection.add(layer) if s else layers.selection.discard(layer)
    layers.remove_selected()
    assert view.model().rowCount() == len(layers)
    assert check_layout_layers(view, layers)


def test_clearing_layerlist(qtbot):
    """Test clearing layer list."""
    layers = LayerList()
    view = QtLayerList(layers)

    qtbot.addWidget(view)

    layers.extend([Image(np.random.random((15, 15))) for _ in range(4)])

    assert view.model().rowCount() == len(layers)
    assert check_layout_layers(view, layers)

    layers.clear()
    assert len(layers) == 0
    assert view.model().rowCount() == len(layers)
    assert check_layout_layers(view, layers)


def test_reordering_layers(qtbot):
    """
    Test reordering layers.
    """
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
    layers[:] = [layers[i] for i in (1, 0, 3, 2)]
    assert view.model().rowCount() == len(layers)
    assert check_layout_layers(view, layers)

    # Do another reorder and check layout and layers list match
    # after swapping layers again
    layers[:] = [layers[i] for i in (1, 0, 3, 2)]
    assert view.model().rowCount() == len(layers)
    assert check_layout_layers(view, layers)

    # Check layout and layers list match after reversing list
    layers.reverse()
    assert view.model().rowCount() == len(layers)
    assert check_layout_layers(view, layers)

    # Check layout and layers list match after rearranging selected layers
    layer_e = Image(np.random.random((15, 15)))
    layer_f = Image(np.random.random((15, 15)))
    layers.append(layer_e)
    layers.append(layer_f)
    for layer, s in zip(layers, [False, True, False, False, True, False]):
        layers.selection.add(layer) if s else layers.selection.discard(layer)
    layers.move_selected(1, 2)
    assert view.model().rowCount() == len(layers)
    assert check_layout_layers(view, layers)


def test_hex_to_name_is_updated():
    fail_msg = (
        "If this test fails then vispy have probably updated their color dictionary, located "
        "in vispy.color.get_color_dict. This not necessarily a bad thing, but make sure that "
        "nothing terrible has happened due to this change."
    )
    new_hex_to_name = {
        f"{v.lower()}ff": k for k, v in get_color_dict().items()
    }
    new_hex_to_name["#00000000"] = 'transparent'
    assert new_hex_to_name == hex_to_name, fail_msg
