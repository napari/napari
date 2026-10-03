import numpy as np
import pytest

from napari._vispy.overlays.text import VispyTextOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari._vispy.visuals.text import Text
from napari.components import ViewerModel
from napari.components.overlays import TextOverlay


@pytest.mark.usefixtures('qapp')
def test_text_instantiation():
    viewer = ViewerModel()
    model = TextOverlay()
    VispyTextOverlay(overlay=model, viewer=viewer, font_info=FontInfo())


@pytest.mark.usefixtures('qapp')
def test_text_width_scales_with_font_size():
    # the box around the text must match the rendered glyphs, which are
    # scaled from one high-res font, so the size is linear in the font size
    small = Text(text='nuclei', font_size=10)
    large = Text(text='nuclei', font_size=20)
    np.testing.assert_allclose(
        np.array(large.get_width_height()),
        2 * np.array(small.get_width_height()),
    )
    np.testing.assert_allclose(
        large.get_line_height(), 2 * small.get_line_height()
    )
