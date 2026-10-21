import numpy as np
import pytest
from pydantic import ValidationError

from napari.components.overlays import CrosshairOverlay


def test_crosshair_overlay_defaults():
    """Test the default state of the crosshair overlay"""
    crosshair = CrosshairOverlay()
    assert not crosshair.visible
    assert crosshair.gap == 20
    assert crosshair.opacity == pytest.approx(0.7)
    np.testing.assert_allclose(crosshair.color, [1, 0, 0, 1])


def test_crosshair_overlay_in_scene():
    """Test the crosshair is exposed by the viewer scene overlays"""
    from napari.components import ViewerModel

    viewer = ViewerModel()
    assert isinstance(viewer.scene.overlays.crosshair, CrosshairOverlay)
    assert not viewer.scene.overlays.crosshair.visible


@pytest.mark.parametrize('gap', [0, -1])
def test_crosshair_gap_must_be_positive(gap):
    """Test the gap of the crosshair cannot be zero or negative"""
    with pytest.raises(ValidationError):
        CrosshairOverlay(gap=gap)
