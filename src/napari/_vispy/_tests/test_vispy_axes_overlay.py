from napari._vispy.overlays.axes import VispyAxesOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import AxesOverlay


def test_floating_axes_dimensions_properly_detected():
    viewer = ViewerModel()
    axes_model = AxesOverlay()
    axes_view = VispyAxesOverlay(
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
