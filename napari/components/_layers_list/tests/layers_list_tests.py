from napari.components import LayersList


def test_layers_list():
    """
    Tests instantiating an empty layers list object
    """
    layers = LayersList()

    assert len(layers) == 0
