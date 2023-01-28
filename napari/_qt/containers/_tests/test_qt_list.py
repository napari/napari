from unittest.mock import Mock

import pytest
from qtpy.QtCore import QEvent, QModelIndex, Qt
from qtpy.QtGui import QKeyEvent

from napari._qt.containers import QtListModel, QtListView
from napari.utils.events._tests.test_evented_list import BASIC_INDICES
from napari.utils.events.containers import SelectableEventedList


class T:
    def __init__(self, name) -> None:
        self.name = name

    def __str__(self):
        return str(self.name)

    def __hash__(self):
        return id(self)

    def __eq__(self, o: object) -> bool:
        return self.name == o


def test_list_model():
    root: SelectableEventedList[str] = SelectableEventedList('abcdef')
    model = QtListModel(root)
    assert all(
        model.data(model.index(i), Qt.UserRole) == letter
        for i, letter in enumerate('abcdef')
    )
    assert all(
        model.data(model.index(i), Qt.DisplayRole) == letter
        for i, letter in enumerate('abcdef')
    )
    # unknown data role
    assert not any(model.data(model.index(i), Qt.FontRole) for i in range(5))
    assert model.flags(QModelIndex()) & Qt.ItemIsDropEnabled
    assert not (model.flags(model.index(1)) & Qt.ItemIsDropEnabled)

    with pytest.raises(TypeError):
        model.setRoot('asdf')

    # smoke test that we can change the root model.
    model.setRoot(SelectableEventedList('zysv'))


def test_list_view(qtbot):
    root: SelectableEventedList[T] = SelectableEventedList(map(T, range(5)))
    root.selection.clear()
    assert not root.selection
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
    root.selection._current = root[3]
    # check current in Qt
    assert root.selection._current == root[3]
    assert qmodel.getItem(qsel.currentIndex()) == root[3]

    # clear current in Qt
    qsel.setCurrentIndex(QModelIndex(), qsel.SelectionFlag.Current)
    # check current in python
    assert root.selection._current is None


def test_list_view_keypress(qtbot):
    root: SelectableEventedList[T] = SelectableEventedList(map(T, range(5)))
    view = QtListView(root)
    qtbot.addWidget(view)

    first = root[0]
    root.selection = {first}
    assert first in root.selection
    # delete removes the item from the python model too
    view.keyPressEvent(
        QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)
    )
    assert first not in root


@pytest.mark.parametrize('sources, dest, expectation', BASIC_INDICES)
def test_move_multiple(sources, dest, expectation):
    """Test that models stay in sync with complicated moves.

    This uses mimeData to simulate drag/drop operations.
    """
    root = SelectableEventedList(map(T, range(8)))
    root.events = Mock(wraps=root.events)
    assert root != expectation

    qt_tree = QtListModel(root)
    dest_mi = qt_tree.index(dest)
    qt_tree.dropMimeData(
        qt_tree.mimeData([qt_tree.index(i) for i in sources]),
        Qt.MoveAction,
        dest_mi.row(),
        dest_mi.column(),
        dest_mi.parent(),
    )
    assert root == qt_tree._root == expectation

    root.events.moving.assert_called()
    root.events.moved.assert_called()
    root.events.reordered.assert_called_with(value=[T(i) for i in expectation])
