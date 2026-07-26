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

    dims.order = (0, 1)
    assert tuple(dim_sorter.axis_list) == (0, 1)
    dim_sorter.axis_list.move(1, 0)
    assert tuple(dim_sorter.axis_list) == (1, 0)
    assert tuple(dims.order) == (1, 0)


def test_dims_sorter_disabled_while_navigation_locked(qtbot):
    """The sorter reorders axes via a direct ``dims.order`` assignment the
    navigation lock cannot guard, so the view is disabled while locked (and
    re-enabled on unlock), mirroring QtDims disabling the slice sliders."""
    dims = Dims(ndim=3)
    parent = QWidget()
    dim_sorter = QtDimsSorter(dims, parent)
    qtbot.addWidget(dim_sorter)
    assert dim_sorter.view.isEnabled()

    token = object()
    dims.lock_navigation(token)
    assert not dim_sorter.view.isEnabled()

    dims.unlock_navigation(token)
    assert dim_sorter.view.isEnabled()


def test_dims_sorter_disabled_when_created_under_lock(qtbot):
    """A sorter created while navigation is already locked starts disabled."""
    dims = Dims(ndim=3)
    dims.lock_navigation(object())
    parent = QWidget()
    dim_sorter = QtDimsSorter(dims, parent)
    qtbot.addWidget(dim_sorter)
    assert not dim_sorter.view.isEnabled()


def test_dims_sorter_callback_management(qtbot):
    dims = Dims()
    parent = QWidget()
    base_callback_count = len(dims.events.order.callbacks)
    dim_sorter = QtDimsSorter(dims, parent)
    qtbot.addWidget(dim_sorter)

    # assert callback hook up
    assert len(dims.events.order.callbacks) == base_callback_count + 1
    assert len(dim_sorter.axis_list.events.reordered.callbacks) == 2

    # Change dims order to trigger axis_list recreation
    # then test that a fresh callback is added
    dims.order = (1, 0)
    assert len(dim_sorter.axis_list.events.reordered.callbacks) == 2


def test_dims_sorter_with_reordered_init(qtbot):
    dims = Dims()
    parent = QWidget()
    dims.order = (1, 0)
    dim_sorter = QtDimsSorter(dims, parent)
    qtbot.addWidget(dim_sorter)
    assert tuple(dim_sorter.axis_list) == tuple(dims.order)
