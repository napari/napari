import pytest

from napari._vispy.overlays.text import VispyTextOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import TextOverlay


@pytest.mark.usefixtures('qapp')
def test_text_instantiation():
    viewer = ViewerModel()
    model = TextOverlay()
    VispyTextOverlay(overlay=model, viewer=viewer, font_info=FontInfo())
