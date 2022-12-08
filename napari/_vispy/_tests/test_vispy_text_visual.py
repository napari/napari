from napari._vispy.overlays.text import VispyTextOverlay
from napari.components.overlays import TextOverlay


def test_text(make_napari_viewer):
    viewer = make_napari_viewer()
    model = TextOverlay()
    visual = VispyTextOverlay(overlay=model, viewer=viewer)

    model.visible = False
    assert visual.node.visible is False
