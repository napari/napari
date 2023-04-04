import numpy as np

import napari.layers


def test_add_function_widget(make_napari_viewer):
    """Test basic add_function_widget functionality"""
    from qtpy.QtWidgets import QDockWidget

    viewer = make_napari_viewer()

    # Define a function.
    def image_sum(
        layerA: napari.layers.Image, layerB: napari.layers.Image
    ) -> napari.layers.Image:
        """Add two layers."""
        if layerA is not None and layerB is not None:
            return napari.layers.Image(layerA.data + layerB.data)
        return None

    dwidg = viewer.window.add_function_widget(image_sum)
    assert dwidg.name == 'image sum'
    assert viewer.window._qt_window.findChild(QDockWidget, 'image sum')

    # make sure that the choice of layers stays in sync with viewer.layers
    _magic_widget = dwidg.widget()._magic_widget
    assert _magic_widget.layerA.choices == ()
    layer = viewer.add_image(np.random.rand(10, 10))
    assert layer in _magic_widget.layerA.choices
