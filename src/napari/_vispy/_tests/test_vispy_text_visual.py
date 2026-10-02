import numpy as np
import pytest

from napari._vispy.overlays.text import (
    VispyLayerNameOverlay,
    VispyTextOverlay,
)
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import TextOverlay
from napari.layers import Image


@pytest.mark.usefixtures('qapp')
def test_text_instantiation():
    viewer = ViewerModel()
    model = TextOverlay()
    VispyTextOverlay(overlay=model, viewer=viewer, font_info=FontInfo())


@pytest.mark.usefixtures('qapp')
def test_layer_name_overlay_is_bold():
    viewer = ViewerModel()
    layer = Image(np.zeros((2, 2)), name='nuclei')
    name_overlay = VispyLayerNameOverlay(
        layer=layer,
        overlay=layer.name_overlay,
        viewer=viewer,
        font_info=FontInfo(),
    )
    text_overlay = VispyTextOverlay(
        overlay=TextOverlay(text='nuclei'),
        viewer=viewer,
        font_info=FontInfo(),
    )
    assert name_overlay.node.bold
    assert not text_overlay.node.bold
    # the size used for tiling and the box accounts for the bold glyphs
    assert name_overlay.x_size > text_overlay.x_size

    layer.name_overlay.bold = False
    assert not name_overlay.node.bold
    assert name_overlay.x_size == text_overlay.x_size
