import numpy as np

from napari._vispy.overlays.scale_bar import VispyScaleBarOverlay
from napari.components.overlays import ScaleBarOverlay


def test_scale_bar_instantiation(viewer_model):
    img = viewer_model.add_image(data=np.zeros((10, 10)), name='test')
    model = ScaleBarOverlay()
    vispy_scale_bar = VispyScaleBarOverlay(overlay=model, viewer=viewer_model)
    assert vispy_scale_bar.overlay.length is None
    model.length = 50
    assert vispy_scale_bar.overlay.length == 50
    assert vispy_scale_bar._unit == viewer_model.layers.units[-1]

    img.units = ('um', 'um')
    assert vispy_scale_bar._unit == viewer_model.layers.units[-1]
