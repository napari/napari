from napari._vispy.overlays.scale_bar import VispyScaleBarOverlay
from napari.components.overlays import ScaleBarOverlay


def test_scale_bar_instantiation(make_napari_viewer):
    viewer = make_napari_viewer()
    model = ScaleBarOverlay()
    model_fixed = ScaleBarOverlay(fixed_width=50)
    vispy_scale_bar = VispyScaleBarOverlay(overlay=model, viewer=viewer)
    vispy_scale_bar_fixed = VispyScaleBarOverlay(
        overlay=model_fixed, viewer=viewer
    )

    assert vispy_scale_bar.overlay.fixed_width is None
    assert vispy_scale_bar_fixed.overlay.fixed_width == 50
