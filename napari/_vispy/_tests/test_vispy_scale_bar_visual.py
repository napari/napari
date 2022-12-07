from napari._vispy.overlays.scale_bar import VispyScaleBarOverlay
from napari.components.overlays import ScaleBarOverlay


def test_scale_bar(make_napari_viewer):
    viewer = make_napari_viewer()
    model = ScaleBarOverlay()
    visual = VispyScaleBarOverlay(overlay=model, viewer=viewer)

    assert visual.node.box.visible is False
    model.box = True
    assert visual.node.box.visible is True
