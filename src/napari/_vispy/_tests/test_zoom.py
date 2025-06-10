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


def test_zoom_overlay_2d():
    zoom_model = ZoomOverlay()
    zoom_model.bounds = ((0, 0), (10, 10))
    d1_min, d1_max, d2_min, d2_max, d3_min, d3_max = zoom_model.extents((0, 1))
    assert d1_min == 0
    assert d1_max == 10
    assert d2_min == 0
    assert d2_max == 10
    assert d3_min == 1
    assert d3_max == 1


def test_zoom_overlay_3d():
    zoom_model = ZoomOverlay()
    zoom_model.bounds = ((0, 0, 0), (20, 10, 10))
    d1_min, d1_max, d2_min, d2_max, d3_min, d3_max = zoom_model.extents(
        (0, 1, 2)
    )
    assert d1_min == 0
    assert d1_max == 20
    assert d2_min == 0
    assert d2_max == 10
    assert d3_min == 0
    assert d3_max == 10
