from napari.components import LayersList


def test_layers_list():
    """
    Tests instantiating an empty LayersList object
    """
    layers = LayersList()

    assert len(layers) == 0
