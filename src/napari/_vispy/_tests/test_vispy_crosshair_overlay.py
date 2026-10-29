import numpy as np
import pytest

from napari._vispy.overlays.crosshair import VispyCrosshairOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari._vispy.visuals.axes import AXES_COLORS
from napari.components import ViewerModel
from napari.components.overlays import CrosshairOverlay


@pytest.fixture
def crosshair_overlay():
    """A vispy crosshair overlay, with the viewer and the model it follows."""
    viewer = ViewerModel()
    overlay = CrosshairOverlay()
    vispy_overlay = VispyCrosshairOverlay(
        viewer=viewer, overlay=overlay, font_info=FontInfo()
    )
    return viewer, overlay, vispy_overlay


@pytest.mark.parametrize('ndim', [2, 3])
def test_crosshair_follows_cursor(crosshair_overlay, ndim):
    """Test the crosshair is placed at the cursor position"""
    viewer, _, vispy_overlay = crosshair_overlay
    if ndim == 3:
        viewer.add_image(np.zeros((10, 20, 30)))
        viewer.dims.ndisplay = 3
        position = (1.0, 2.0, 3.0)
    else:
        position = (3.0, 4.0)

    viewer.cursor.position = position
    # vispy axes are the reverse of the napari world axes
    np.testing.assert_allclose(
        vispy_overlay.node.position[:ndim], position[::-1]
    )


def test_crosshair_follows_model(crosshair_overlay):
    """Test the crosshair visual is updated when the model changes"""
    _, overlay, vispy_overlay = crosshair_overlay
    node = vispy_overlay.node

    assert not node.visible
    overlay.visible = True
    assert node.visible

    overlay.color = 'blue'
    # both displayed lines are blue, z is unused in 2D
    np.testing.assert_allclose(node.color[:2], [[0, 0, 1, 1]] * 2)

    assert node.gap == 20
    overlay.gap = 35
    assert node.gap == 35


def test_crosshair_updates_on_dims_change(crosshair_overlay):
    """Test the crosshair is not stale after the displayed dims change"""
    viewer, _, vispy_overlay = crosshair_overlay
    viewer.add_image(np.zeros((10, 20, 30)))
    viewer.cursor.position = (1.0, 2.0, 3.0)

    # 2D: the last two axes are displayed, in vispy (x, y) order, z is 0
    np.testing.assert_allclose(vispy_overlay.node.position, [3, 2, 0])

    viewer.dims.ndisplay = 3
    np.testing.assert_allclose(vispy_overlay.node.position, [3, 2, 1])

    viewer.dims.order = (1, 0, 2)
    np.testing.assert_allclose(vispy_overlay.node.position, [3, 1, 2])


def test_crosshair_default_axes_colors(crosshair_overlay):
    """Test each line takes the color of the matching canvas axes line"""
    viewer, overlay, vispy_overlay = crosshair_overlay
    assert overlay.color is None
    viewer.add_image(np.zeros((10, 20, 30)))
    viewer.dims.ndisplay = 3
    cyan, magenta, yellow = AXES_COLORS[:3]
    # vispy order is x, y, z, i.e. the reverse of the napari axes
    np.testing.assert_allclose(
        vispy_overlay.node.color, [cyan, magenta, yellow]
    )

    viewer.dims.order = (1, 0, 2)
    np.testing.assert_allclose(
        vispy_overlay.node.color, [cyan, yellow, magenta]
    )

    viewer.dims.ndisplay = 2
    # only the two displayed axes (0 and 2 here) are colored, z is unused
    np.testing.assert_allclose(vispy_overlay.node.color[:2], [cyan, yellow])


def test_crosshair_default_axes_colors_2d_data_3d_view(crosshair_overlay):
    """Test the z line has its own color for 2D data in a 3D view"""
    viewer, _, vispy_overlay = crosshair_overlay
    viewer.add_image(np.zeros((10, 20)))
    viewer.dims.ndisplay = 3
    cyan, magenta, yellow = AXES_COLORS[:3]
    # same colors as once the data gains a leading dim
    np.testing.assert_allclose(
        vispy_overlay.node.color, [cyan, magenta, yellow]
    )


def test_crosshair_three_colors(crosshair_overlay):
    """Test three colors follow the displayed axes, in napari order"""
    viewer, overlay, vispy_overlay = crosshair_overlay
    viewer.add_image(np.zeros((10, 20, 30)))
    viewer.dims.ndisplay = 3
    red, green, blue = [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]
    overlay.color = ('red', 'lime', 'blue')
    # vispy order is x, y, z, i.e. the reverse of the napari axes
    np.testing.assert_allclose(vispy_overlay.node.color, [blue, green, red])

    # in 2D, only the last two colors are used
    viewer.dims.ndisplay = 2
    np.testing.assert_allclose(vispy_overlay.node.color[:2], [blue, green])
