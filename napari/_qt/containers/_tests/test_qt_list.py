from qtpy.QtCore import QModelIndex

from napari._qt.containers import QtListView
from napari.utils.events.containers import SelectableEventedList


class T:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


def test_list_view(qtbot):
    root: SelectableEventedList[T] = SelectableEventedList(map(T, range(5)))
    view = QtListView(root)
    qmodel = view.model()
    qsel = view.selectionModel()
    qtbot.addWidget(view)

    # update selection in python
    _selection = {root[0], root[2]}
    root.selection.update(_selection)
    assert root[2] in root.selection

    # check selection in Qt
    idx = {qmodel.getItem(i) for i in qsel.selectedIndexes()}
    assert idx == _selection

    # clear selection in Qt
    qsel.clearSelection()
    # check selection in python
    assert not root.selection

    # update current in python
    root.selection.current = root[3]
    # check current in Qt
    assert root.selection.current == root[3]
    assert qmodel.getItem(qsel.currentIndex()) == root[3]

    # clear current in Qt
    qsel.setCurrentIndex(QModelIndex(), qsel.Current)
    # check current in python
    assert root.selection.current is None
