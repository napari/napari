from qtpy.QtWidgets import QWidget

from napari._qt.widgets.qt_dims_sorter import QtDimsSorter
from napari.components.viewer_model import ViewerModel


def test_dims_sorter(qtbot):
    viewer = ViewerModel()
    parent = QWidget()
    dim_sorter = QtDimsSorter(viewer, parent)
    qtbot.addWidget(dim_sorter)
    assert tuple(dim_sorter.axis_list) == (0, 1)

    viewer.dims.axis_labels = ('y', 'x')
    assert tuple(dim_sorter.axis_list) == ('y', 'x')

    dim_sorter.axis_list.move(1, 0)
    assert tuple(dim_sorter.axis_list) == ('x', 'y')
    assert tuple(viewer.dims.order) == (1, 0)


def test_dims_sorter_callback_management(qtbot):
    viewer = ViewerModel()
    parent = QWidget()
    base_callback_count = len(viewer.dims.events.order.callbacks)
    dim_sorter = QtDimsSorter(viewer, parent)
    qtbot.addWidget(dim_sorter)

    # assert callback hook up
    assert len(viewer.dims.events.order.callbacks) == base_callback_count + 1
    assert len(dim_sorter.axis_list.events.reordered.callbacks) == 2

    # assert callback termination
    parent.destroyed.emit()
    assert len(viewer.dims.events.order.callbacks) == base_callback_count
    assert len(dim_sorter.axis_list.events.reordered.callbacks) == 1


def test_dims_sorter_with_reordered_init(qtbot):
    viewer = ViewerModel()
    parent = QWidget()
    viewer.dims.order = (1, 0)
    dim_sorter = QtDimsSorter(viewer, parent)
    qtbot.addWidget(dim_sorter)
    assert tuple(dim_sorter.axis_list) == tuple(viewer.dims.order)
