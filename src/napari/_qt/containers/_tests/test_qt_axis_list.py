from qtpy.QtCore import QModelIndex, Qt

from napari._qt.containers import (
    AxisList,
    AxisModel,
    QtAxisListModel,
    QtListView,
)
from napari.components import Dims

FLAGS = (
    Qt.ItemFlag.ItemIsSelectable
    | Qt.ItemFlag.ItemIsEditable
    | Qt.ItemFlag.ItemIsUserCheckable
    | Qt.ItemFlag.ItemIsEnabled
    | Qt.ItemFlag.ItemNeverHasChildren
    | Qt.ItemFlag.ItemIsDragEnabled
)


def test_axismodel():
    dims = Dims()
    axismodel = AxisModel(dims, 0)
    assert axismodel == 0
    assert axismodel.rollable

    axismodel.rollable = False
    assert not axismodel.rollable
    assert not dims.rollable[0]


def test_AxisList():
    # from list
    dims = Dims()
    axes = [AxisModel(dims, axis) for axis in dims.order]
    axislist = AxisList(axes)
    assert all(axis == idx for idx, axis in enumerate(axislist))

    # from_dims
    axislist = AxisList.from_dims(dims)
    assert len(axislist) == 2
    for idx, axis in enumerate(axislist):
        assert axis == idx


def test_QtAxisListModel_data(qtbot):
    dims, axislist, listview, axislistmodel = make_QtAxisListModel(qtbot)
    assert all(
        axislistmodel.data(
            axislistmodel.index(idx), role=Qt.ItemDataRole.DisplayRole
        )
        == axislist[idx]
        for idx in dims.order
    )
    assert all(
        axislistmodel.data(
            axislistmodel.index(idx),
            role=Qt.ItemDataRole.TextAlignmentRole,
        )
        == Qt.AlignCenter
        for idx in dims.order
    )
    assert all(
        (
            axislistmodel.data(
                axislistmodel.index(idx),
                role=Qt.ItemDataRole.CheckStateRole,
            ),
            axislist[idx].rollable,
        )
        for idx in dims.order
    )

    # setData
    idx = 1
    with qtbot.waitSignal(axislistmodel.dataChanged, timeout=100):
        assert axislistmodel.setData(
            axislistmodel.index(idx),
            Qt.CheckState.Checked,
            role=Qt.ItemDataRole.CheckStateRole,
        )
    assert dims.rollable[idx]

    new_name = 'new_name'
    with qtbot.waitSignal(axislistmodel.dataChanged, timeout=100):
        assert axislistmodel.setData(
            axislistmodel.index(idx), new_name, role=Qt.ItemDataRole.EditRole
        )
    assert dims.axis_labels[idx] == new_name


def test_QtAxisListModel_flags(qtbot):
    dims, axislist, listview, axislistmodel = make_QtAxisListModel(qtbot)
    assert axislistmodel.flags(QModelIndex()) == Qt.ItemFlag.ItemIsDropEnabled
    flags = [
        axislistmodel.flags(axislistmodel.index(idx)) for idx in dims.order
    ]
    ref_flags = [FLAGS for idx in dims.order]
    assert flags == ref_flags


def make_QtAxisListModel(qtbot) -> tuple[Dims, AxisList, QtAxisListModel]:
    dims = Dims()
    dims.rollable = [True, False]
    axislist = AxisList.from_dims(dims)
    view = QtListView(axislist)
    axislistmodel = view.model()
    return dims, axislist, view, axislistmodel
