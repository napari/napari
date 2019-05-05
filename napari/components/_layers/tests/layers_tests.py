from napari.components import Layers


def test_layers():
    """
    Tests instantiating an empty layers object
    """
    layers = Layers()

    assert len(layers) == 0
