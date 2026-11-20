import numpy as np
import pytest

from napari._vispy.overlays.colorbar import VispyColorBarOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.layers import Tracks
from napari.utils.colormaps import Colormap

# two tracks with three points each: ID, T, Y, X
TRACKS_DATA = np.array(
    [[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2]]
    + [[1, 0, 5, 5], [1, 1, 6, 6], [1, 2, 7, 7]]
)
SPEED = np.array([0, 2, 4, 6, 8, 10])


def _make_overlay(layer):
    return VispyColorBarOverlay(
        layer=layer,
        overlay=layer.colorbar,
        viewer=ViewerModel(),
        font_info=FontInfo(),
    )


@pytest.mark.usefixtures('qapp')
def test_tracks_colorbar_follows_color_by_and_colormap():
    layer = Tracks(TRACKS_DATA, features={'speed': SPEED}, color_by='speed')
    vispy_overlay = _make_overlay(layer)
    np.testing.assert_allclose(vispy_overlay.node.img.clim, (0, 10))
    assert vispy_overlay.source_wrapper.colormap.name == 'turbo'

    layer.colormap = 'viridis'
    assert vispy_overlay.source_wrapper.colormap.name == 'viridis'

    layer.features = {'speed': SPEED * 2}
    np.testing.assert_allclose(vispy_overlay.node.img.clim, (0, 20))

    layer.color_by = 'track_id'
    np.testing.assert_allclose(vispy_overlay.node.img.clim, (0, 1))


@pytest.mark.usefixtures('qapp')
def test_tracks_colorbar_with_colormaps_dict():
    """Colormaps from colormaps_dict get the values without normalization."""
    cmap = Colormap([[1, 0, 0, 1], [0, 0, 1, 1]], name='red_blue')
    layer = Tracks(
        TRACKS_DATA,
        features={'speed': SPEED},
        color_by='speed',
        colormaps_dict={'speed': cmap},
    )
    vispy_overlay = _make_overlay(layer)
    np.testing.assert_allclose(vispy_overlay.node.img.clim, (0, 1))
    assert vispy_overlay.source_wrapper.colormap is cmap
