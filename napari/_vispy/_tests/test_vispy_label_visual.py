"""Test label visual"""


def test_vispy_label_visual_image(make_napari_viewer):
    viewer = make_napari_viewer()
    qt_widget = viewer.window.qt_viewer
    assert viewer.label is not None
    # check font size attribute
    assert qt_widget.label.node.font_size == viewer.label.font_size
    viewer.label.font_size = 13
    assert qt_widget.label.node.font_size == viewer.label.font_size == 13
    # check text attribute
    assert qt_widget.label.node.text == viewer.label.text
    viewer.label.text = "TEST TEXT"
    assert qt_widget.label.node.text == viewer.label.text == "TEST TEXT"
    # check visible attribute
    assert qt_widget.label.node.visible == viewer.label.visible
    viewer.label.visible = True
    assert qt_widget.label.node.visible == viewer.label.visible
