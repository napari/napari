import os
import sys

import numpy as np
import pytest

from napari.layers import Image, Labels, Layer, Points

magicgui = pytest.importorskip('magicgui', reason='please install magicgui.')


def test_magicgui_returns_image(make_test_viewer):
    """make sure a magicgui function returning Image adds an Image."""
    viewer = make_test_viewer()

    @magicgui.magicgui
    def add_image() -> Image:
        return np.random.rand(10, 10)

    viewer.window.add_dock_widget(add_image)
    assert len(viewer.layers) == 0
    add_image()  # should add a new layer to the list
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == 'add_image result'

    add_image()  # should just update existing layer on subsequent calls
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == 'add_image result'
    assert isinstance(viewer.layers[0], Image)


def test_magicgui_returns_label(make_test_viewer):
    """make sure a magicgui function returning Labels adds a Labels."""
    viewer = make_test_viewer()

    @magicgui.magicgui
    def add_labels() -> Labels:
        return np.random.rand(10, 10)

    viewer.window.add_dock_widget(add_labels)
    assert len(viewer.layers) == 0
    add_labels()  # should add a new layer to the list
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == 'add_labels result'
    assert isinstance(viewer.layers[0], Image)


@pytest.mark.skipif(
    bool(os.environ.get('CI') and sys.platform == "darwin"),
    reason="segfault on mac CI",
)
def test_magicgui_returns_layer_tuple(make_test_viewer):
    """make sure a magicgui function returning Layer adds the right type."""
    viewer = make_test_viewer()

    @magicgui.magicgui
    def add_layer() -> Layer:
        return [(np.random.rand(10, 3), {'size': 20, 'name': 'foo'}, 'points')]

    viewer.window.add_dock_widget(add_layer)
    assert len(viewer.layers) == 0

    add_layer()  # should add a new layer to the list
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == 'foo'
    assert isinstance(layer, Points)
    assert layer.data.shape == (10, 3)
