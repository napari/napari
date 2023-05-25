from napari._vispy.overlays.text import VispyTextOverlay
from napari.components.overlays import TextOverlay


def test_text_instantiation(make_napari_viewer):
    viewer = make_napari_viewer()
    model = TextOverlay()
    VispyTextOverlay(overlay=model, viewer=viewer)
