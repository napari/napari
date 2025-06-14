from napari._vispy.overlays.zoom import VispyZoomOverlay
from napari.components.overlays import ZoomOverlay


def test_zoom_overlay_initialization(make_napari_viewer):
    viewer = make_napari_viewer()
    zoom_model = ZoomOverlay()
    zoom_view = VispyZoomOverlay(viewer=viewer, overlay=zoom_model)

    # change the visibility
    assert zoom_view.node is not None
    assert zoom_view.node.visible is False
    zoom_model.visible = True
    assert zoom_view.node.visible is True
