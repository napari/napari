import os
from napari.components import LayersList
from napari.components._layers_list.view import QtLayersList, QtDivider
from napari.util import app_context
from napari.layers import Image
import numpy as np

os.environ['NAPARI_TEST'] = '1'


def test_divider():
    """
    Test creating the divider.
    """
    with app_context():
        divider = QtDivider()

        # Check divider was created properly
        assert type(divider) == QtDivider

        # Check divider property defaults to False
        assert divider.property('selected') == False

        # Set divider property
        divider.setSelected(True)
        assert divider.property('selected') == True
        divider.setSelected(False)
        assert divider.property('selected') == False


def test_creating_empty_view():
    """
    Test creating LayerList view.
    """
    with app_context():
        layers = LayersList()
        view = QtLayersList(layers)

        # Check that the layers model has been appended to the layers view
        assert view.layers == layers

        # Check that vbox_layout only contains one QtDivider and one spacer
        assert view.vbox_layout.count() == 2
        assert type(view.vbox_layout.itemAt(0).widget()) == QtDivider


def test_adding_layers():
    """
    Test creating LayerList view.
    """
    with app_context():
        layers = LayersList()
        view = QtLayersList(layers)

        # Check that new layer and divider get added to vbox_layout
        layer_a = Image(np.random.random((10, 10)))
        layers.append(layer_a)
        assert view.vbox_layout.count() == 4
        assert [
            type(view.vbox_layout.itemAt(2 * i).widget()) for i in range(2)
        ] == [QtDivider] * 2
        assert view.vbox_layout.itemAt(1).widget().layer == layer_a

        # Check that new layers and divider get added to vbox_layout
        layer_b = Image(np.random.random((15, 15)))
        layer_c = Image(np.random.random((15, 15)))
        layer_d = Image(np.random.random((15, 15)))
        layers.append(layer_b)
        layers.append(layer_c)
        layers.append(layer_d)
        assert view.vbox_layout.count() == (1 + len(layers)) * 2
        assert [
            type(view.vbox_layout.itemAt(2 * i).widget())
            for i in range(1 + len(layers))
        ] == [QtDivider] * (1 + len(layers))
        assert [
            view.vbox_layout.itemAt(2 * i - 1).widget().layer
            for i in range(len(layers), 0, -1)
        ] == [layer_a, layer_b, layer_c, layer_d]


# def test_creating_view():
#     """
#     Test creating LayerList view.
#     """
#     with app_context():
#         layers = LayersList()
#         view = QtLayersList(layers)
#
#         # Check that the layers model has been appended to the layers view
#         assert view.layers == layers
#
#         # Check that vbox_layout only contains one QtDivider and one spacer
#         assert view.vbox_layout.count() == 2
#         assert type(view.vbox_layout.itemAt(0).widget()) == QtDivider

# def test_reordering():
#     """
#     Test reordering LayerList view.
#     """
#     with app_context():
#         layers = LayersList()
#         view = QtLayersList(layers)
#
#         # Check that the layers model has been appended to the layers view
#         for self.vbox_layout.itemAt(i + 1).widget()
#         assert view.layers == layers
