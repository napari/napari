from qtpy.QtWidgets import QWidget

from napari._qt.widgets.qt_dims_sorter import QtDimsSorter
from napari.components.dims import Dims


def test_dims_sorter(qtbot):
    dims = Dims()
    parent = QWidget()
    dim_sorter = QtDimsSorter(dims, parent)
    qtbot.addWidget(dim_sorter)
    assert tuple(dim_sorter.axis_list) == (0, 1)

    dims.axis_labels = ('y', 'x')
    assert tuple(dim_sorter.axis_list) == ('y', 'x')

    dim_sorter.axis_list.move(1, 0)
    assert tuple(dim_sorter.axis_list) == ('x', 'y')
    assert tuple(dims.order) == (1, 0)


def test_dims_sorter_callback_management(qtbot):
    dims = Dims()
    parent = QWidget()
    base_callback_count = len(dims.events.order.callbacks)
    dim_sorter = QtDimsSorter(dims, parent)
    qtbot.addWidget(dim_sorter)

    # assert callback hook up
    assert len(dims.events.order.callbacks) == base_callback_count + 1
    assert len(dim_sorter.axis_list.events.reordered.callbacks) == 2


def test_dims_sorter_with_reordered_init(qtbot):
    dims = Dims()
    parent = QWidget()
    dims.order = (1, 0)
    dim_sorter = QtDimsSorter(dims, parent)
    qtbot.addWidget(dim_sorter)
    assert tuple(dim_sorter.axis_list) == tuple(dims.order)
