"""Test label visual"""
import pytest

from napari.components._viewer_constants import TextOverlayPosition


def test_vispy_text_visual(make_napari_viewer):
    viewer = make_napari_viewer()
    qt_widget = viewer.window.qt_viewer
    assert viewer.text_overlay is not None
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
    for position in list(TextOverlayPosition):
        viewer.text_overlay.position = position
        assert viewer.text_overlay.position == position
    with pytest.raises(ValueError):
        viewer.text_overlay.position = "top_centre"
