from napari._vispy.overlays.zoom import VispyRectangleSelectOverlay
from napari.components.overlays import RectangleSelectOverlay
from napari.components.viewer_model import ViewerModel


def test_zoom_overlay_initialization():
    viewer = ViewerModel()
    zoom_model = RectangleSelectOverlay()
    zoom_view = VispyRectangleSelectOverlay(viewer=viewer, overlay=zoom_model)

    # change the visibility
    assert zoom_view.node is not None
    assert zoom_view.node.visible is False
    zoom_model.visible = True
    assert zoom_view.node.visible is True
