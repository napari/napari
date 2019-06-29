import os
from napari.components import LayersList
from napari.components._layers_list.view import QtLayersList, QtDivider
from napari.util import app_context

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


def test_creating_view():
    """
    Test creating LayerList view.
    """
    with app_context():
        layers = LayersList()
        view = QtLayersList(layers)

        # Check that the layers model has been appended to the layers view
        assert view.layers == layers
