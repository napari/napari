from napari._vispy.overlays.scale_bar import VispyScaleBarOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import ScaleBarOverlay


def test_scale_bar_instantiation():
    viewer = ViewerModel()
    model = ScaleBarOverlay()
    font_info = FontInfo()
    vispy_scale_bar = VispyScaleBarOverlay(
        overlay=model, viewer=viewer, font_info=font_info
    )
    assert vispy_scale_bar.overlay.length is None
    model.length = 50
    assert vispy_scale_bar.overlay.length == 50
    assert vispy_scale_bar.overlay.unit == 'pixel'
