from napari._qt.widgets.qt_dims_sorter import QtDimsSorter
from napari.components.viewer_model import ViewerModel


def test_dims_sorter(qtbot):
    viewer = ViewerModel()
    dim_sorter = QtDimsSorter(viewer)
    qtbot.addWidget(dim_sorter)
    assert tuple(dim_sorter.axes_list) == (0, 1)

    viewer.dims.axis_labels = ('y', 'x')
    assert tuple(dim_sorter.axes_list) == ('y', 'x')

    dim_sorter.axes_list.move(1, 0)
    assert tuple(dim_sorter.axes_list) == ('x', 'y')
    assert tuple(viewer.dims.order) == (1, 0)


def test_dims_sorter_with_reordered_init(qtbot):
    viewer = ViewerModel()
    viewer.dims.order = (1, 0)

    dim_sorter = QtDimsSorter(viewer)
    qtbot.addWidget(dim_sorter)
    assert tuple(dim_sorter.axes_list) == tuple(viewer.dims.order)
