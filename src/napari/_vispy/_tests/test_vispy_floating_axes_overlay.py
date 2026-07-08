import numpy as np
from vispy.util.quaternion import Quaternion

from napari._vispy.overlays.floating_axes import VispyFloatingAxesOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import FloatingAxesOverlay


def test_floating_axes_dimensions_properly_detected():
    viewer = ViewerModel()
    axes_model = FloatingAxesOverlay()
    axes_view = VispyFloatingAxesOverlay(
        viewer=viewer, overlay=axes_model, font_info=FontInfo()
    )
    viewer.dims.ndim = 2
    viewer.dims.ndisplay = 3
    assert tuple(axes_view.node.axes.text.text) == ('-1', '-2')

    viewer.dims.ndim = 3
    viewer.dims.ndisplay = 2
    assert tuple(axes_view.node.axes.text.text) == ('-1', '-2')

    viewer.dims.ndim = 2
    viewer.dims.ndisplay = 3
    assert tuple(axes_view.node.axes.text.text) == ('-1', '-2')

    viewer.dims.ndim = 3
    viewer.dims.ndisplay = 3
    assert tuple(axes_view.node.axes.text.text) == ('-1', '-2', '-3')


def _assert_quat_equal(q1, q2):
    # vispy quat doesn't have __eq__; some variability
    # is expected due to float imprecision
    np.testing.assert_allclose(
        [q1.x, q1.y, q1.z, q1.w],
        [q2.x, q2.y, q2.z, q2.w],
        rtol=0.005,
    )


def test_angles():
    viewer = ViewerModel()
    axes_model = FloatingAxesOverlay()
    axes_view = VispyFloatingAxesOverlay(
        viewer=viewer, overlay=axes_model, font_info=FontInfo()
    )
    viewer.dims.ndim = 3
    viewer.dims.ndisplay = 3

    viewer.camera.angles = (0, 0, 0)

    _assert_quat_equal(
        axes_view.node.camera._quaternion, Quaternion(0.707, 0.707, 0, 0)
    )

    viewer.camera.angles = (-45, 0, -45)

    _assert_quat_equal(
        axes_view.node.camera._quaternion,
        Quaternion(0.354, 0.854, -0.146, -0.354),
    )

    # we have z flip by default
    assert axes_view.node.camera.flip == (False, False, True)
    # changing flip affects flip and quaternion
    viewer.camera.orientation = ('away', 'down', 'right')
    axes_view.node.camera.flip = (False, False, False)
    _assert_quat_equal(
        axes_view.node.camera._quaternion,
        Quaternion(0.854, 0.354, -0.354, -0.146),
    )
