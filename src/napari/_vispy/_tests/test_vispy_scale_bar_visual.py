import numpy as np
import pytest

from napari._vispy.overlays.scale_bar import VispyScaleBarOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import ScaleBarOverlay


def test_scale_bar_instantiation(viewer_model: ViewerModel):
    img = viewer_model.add_image(data=np.zeros((10, 10)), name='test')
    model = ScaleBarOverlay()
    font_info = FontInfo()
    vispy_scale_bar = VispyScaleBarOverlay(
        overlay=model, viewer=viewer_model, font_info=font_info
    )
    assert vispy_scale_bar.overlay.length is None
    model.length = 50
    assert vispy_scale_bar.overlay.length == 50
    assert vispy_scale_bar._unit.units == viewer_model.layers.units[-1]

    img.units = ('um', 'um')
    vispy_scale_bar._on_unit_change()
    assert vispy_scale_bar._unit.units == viewer_model.layers.units[-1]


def test_scale_bar_inconsistent_units_default_to_pixel(
    viewer_model: ViewerModel,
):
    img1 = viewer_model.add_image(data=np.zeros((10, 10)), name='test1')
    _img2 = viewer_model.add_image(
        data=np.zeros((10, 10)), name='test2', units=('um', 'um')
    )
    model = ScaleBarOverlay()
    vispy_scale_bar = VispyScaleBarOverlay(
        overlay=model, viewer=viewer_model, font_info=FontInfo()
    )
    # dimensionless when inconsistent
    assert vispy_scale_bar._unit.units.dimensionless
    img1.units = ('um', 'um')
    vispy_scale_bar._on_unit_change()
    assert vispy_scale_bar._unit.units == 'micrometer'
    with pytest.warns(
        FutureWarning,
        match='Setting unit on the ScaleBar model is deprecated.',
    ):
        model.unit = 's'
    vispy_scale_bar._on_unit_change()
    # this has no effect now, it should not change
    assert vispy_scale_bar._unit.units == 'micrometer'
