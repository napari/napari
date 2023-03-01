from typing import TYPE_CHECKING

import numpy as np
from magicgui import magicgui

if TYPE_CHECKING:
    import typing

    import napari.types


def func() -> 'napari.types.ImageData':
    return np.zeros((10, 10))


def func_optional(a: bool) -> 'typing.Optional[napari.types.ImageData]':
    if a:
        return np.zeros((10, 10))
    return None


def test_add_layer_data_to_viewer(make_napari_viewer):
    viewer = make_napari_viewer()
    gui = magicgui(func)
    viewer.window.add_dock_widget(gui)
    assert not viewer.layers

    gui()

    assert len(viewer.layers) == 1


def test_add_layer_data_to_viewer_optional(make_napari_viewer):
    viewer = make_napari_viewer()
    gui = magicgui(func_optional)
    viewer.window.add_dock_widget(gui)
    assert not viewer.layers

    gui(a=True)

    assert len(viewer.layers) == 1

    gui(a=False)

    assert len(viewer.layers) == 1
