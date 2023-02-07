from napari._vispy.overlays.axes import VispyAxesOverlay
from napari.components import ViewerModel
from napari.components.overlays import AxesOverlay


def test_init_with_2d_display_of_2_dimensions():
    axes_model = AxesOverlay()
    viewer = ViewerModel(axis_labels=('a', 'b'))
    viewer.dims.ndim = 2
    viewer.dims.ndisplay = 3

    axes_view = VispyAxesOverlay(viewer=viewer, overlay=axes_model)

    assert tuple(axes_view.node.text.text) == ('b', 'a')


def test_init_with_2d_display_of_3_dimensions():
    axes_model = AxesOverlay()
    viewer = ViewerModel(axis_labels=('a', 'b', 'c'))
    viewer.dims.ndim = 2
    viewer.dims.ndisplay = 3

    axes_view = VispyAxesOverlay(viewer=viewer, overlay=axes_model)

    assert tuple(axes_view.node.text.text) == ('c', 'b')


def test_init_with_3d_display_of_3_dimensions():
    axes_model = AxesOverlay()
    viewer = ViewerModel()
    # TODO: passing through initializer replaces 'a' with '0', likely
    # due to root validator, so set here instead.
    viewer.dims.ndim = 3
    viewer.dims.ndisplay = 3
    viewer.dims.axis_labels = ('a', 'b', 'c')

    axes_view = VispyAxesOverlay(viewer=viewer, overlay=axes_model)

    assert tuple(axes_view.node.text.text) == ('c', 'b', 'a')
