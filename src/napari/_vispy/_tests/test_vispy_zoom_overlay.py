from napari._vispy.overlays.rectangle import VispyViewerRectOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components.overlays import ZoomRectOverlay
from napari.components.viewer_model import ViewerModel


def test_zoom_overlay_initialization():
    viewer = ViewerModel()
    zoom_model = ZoomRectOverlay()
    zoom_view = VispyViewerRectOverlay(
        viewer=viewer, overlay=zoom_model, font_info=FontInfo()
    )

    # change the visibility
    assert zoom_view.node is not None
    assert zoom_view.node.visible is False
    zoom_model.visible = True
    assert zoom_view.node.visible is True
