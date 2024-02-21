from napari._vispy.overlays.slice_text import VispySliceTextOverlay
from napari.components.overlays.slice_text import SliceTextOverlay


def test_slice_text_instantiation(make_napari_viewer):
    viewer = make_napari_viewer()
    model = SliceTextOverlay()
    VispySliceTextOverlay(overlay=model, viewer=viewer)
