from napari._vispy.overlays.scene_axes import VispySceneAxesOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import SceneAxesOverlay


def test_scene_axes_dimensions_properly_detected():
    viewer = ViewerModel()
    axes_model = SceneAxesOverlay()
    axes_view = VispySceneAxesOverlay(
        viewer=viewer, overlay=axes_model, font_info=FontInfo()
    )
    viewer.dims.ndim = 2
    viewer.dims.ndisplay = 3
    assert tuple(axes_view.node.text.text) == ('-1', '-2')

    viewer.dims.ndim = 3
    viewer.dims.ndisplay = 2
    assert tuple(axes_view.node.text.text) == ('-1', '-2')

    viewer.dims.ndim = 2
    viewer.dims.ndisplay = 3
    assert tuple(axes_view.node.text.text) == ('-1', '-2')

    viewer.dims.ndim = 3
    viewer.dims.ndisplay = 3
    assert tuple(axes_view.node.text.text) == ('-1', '-2', '-3')
