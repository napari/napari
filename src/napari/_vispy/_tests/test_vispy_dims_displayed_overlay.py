import numpy as np
import pytest

from napari._vispy.overlays.text import VispyDimsDisplayedOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel


@pytest.mark.usefixtures('qapp')
def test_dims_displayed_overlay_shows_displayed_axes():
    viewer = ViewerModel()
    layer = viewer.add_image(np.zeros((2, 3, 4)), axis_labels=('z', 'y', 'x'))
    vispy_overlay = VispyDimsDisplayedOverlay(
        overlay=viewer.canvas.overlays.dims_displayed,
        viewer=viewer,
        font_info=FontInfo(),
    )
    assert vispy_overlay.node.text == 'y x'

    viewer.dims.order = (1, 0, 2)
    assert vispy_overlay.node.text == 'z x'

    viewer.dims.ndisplay = 3
    assert vispy_overlay.node.text == 'y z x'

    layer.axis_labels = ('t', 'r', 'c')
    assert vispy_overlay.node.text == 'r t c'
