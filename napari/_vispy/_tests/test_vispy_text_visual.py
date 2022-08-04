"""Test label visual."""
import numpy as np
import pytest

from napari.components._viewer_constants import CanvasPosition


def test_vispy_text_visual(make_napari_viewer):
    viewer = make_napari_viewer()
    qt_widget = viewer.window._qt_viewer
    assert viewer.text_overlay is not None
    assert qt_widget.text_overlay is not None

    # check font size attribute
    assert (
        qt_widget.text_overlay.node.font_size == viewer.text_overlay.font_size
    )
    viewer.text_overlay.font_size = 13
    assert (
        qt_widget.text_overlay.node.font_size
        == viewer.text_overlay.font_size
        == 13
    )

    # check text attribute
    assert qt_widget.text_overlay.node.text == viewer.text_overlay.text
    viewer.text_overlay.text = "TEST TEXT"
    assert (
        qt_widget.text_overlay.node.text
        == viewer.text_overlay.text
        == "TEST TEXT"
    )

    # check visible attribute
    assert qt_widget.text_overlay.node.visible == viewer.text_overlay.visible
    viewer.text_overlay.visible = True
    assert qt_widget.text_overlay.node.visible == viewer.text_overlay.visible

    # check position attribute
    for position in list(CanvasPosition):
        viewer.text_overlay.position = position
        assert viewer.text_overlay.position == position
    with pytest.raises(ValueError):
        viewer.text_overlay.position = "top_centre"

    # check color attribute
    for (rgba, color) in [
        ((0.0, 1.0, 1.0, 1.0), "#00ffff"),  # check hex color
        ((1.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),  # check 3 tuple
        ((1.0, 0.5, 0.0, 0.5), (1.0, 0.5, 0.0, 0.5)),  # check 4 tuple
        ((1.0, 1.0, 1.0, 1.0), "white"),  # check text color
    ]:
        viewer.text_overlay.color = color
        np.testing.assert_equal(viewer.text_overlay.color, np.asarray(rgba))
