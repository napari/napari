import pytest
from qtpy.QtCore import QModelIndex, Qt

from napari._qt.containers import QtNodeTreeModel, QtNodeTreeView
from napari._qt.containers._base_item_view import index_of
from napari.utils.events._tests.test_evented_list import NESTED_POS_INDICES
from napari.utils.tree import Group, Node


@pytest.fixture
def tree_model(qapp):
    root = Group(
        [
            Node(name="1"),
            Group(
                [
                    Node(name="2"),
                    Group([Node(name="3"), Node(name="4")], name="g2"),
                    Node(name="5"),
                    Node(name="6"),
                    Node(name="7"),
                ],
                name="g1",
            ),
            Node(name="8"),
            Node(name="9"),
        ],
        name="root",
    )
    return QtNodeTreeModel(root)


def _recursive_make_group(lst, level=0):
    """Make a Tree of Group/Node objects from a nested list."""
    out = []
    for item in lst:
        if isinstance(item, list):
            out.append(_recursive_make_group(item, level=level + 1))
        else:
            out.append(Node(name=str(item)))
    return Group(out, name=f'g{level}')


def _assert_models_synced(model: Group, qt_model: QtNodeTreeModel):
    for item in model.traverse():
        model_idx = qt_model.nestedIndex(item.index_from_root())
        node = qt_model.getItem(model_idx)
        assert item.name == node.name


def test_move_single_tree_item(tree_model):
    """Test moving a single item."""
    root = tree_model._root
    assert isinstance(root, Group)
    _assert_models_synced(root, tree_model)
    root.move(0, 2)
    _assert_models_synced(root, tree_model)
    root.move(3, 1)
    _assert_models_synced(root, tree_model)


@pytest.mark.parametrize('sources, dest, expectation', NESTED_POS_INDICES)
def test_nested_move_multiple(qapp, sources, dest, expectation):
    """Test that models stay in sync with complicated moves.

    This uses mimeData to simulate drag/drop operations.
    """
    root = _recursive_make_group([0, 1, [20, [210, 211], 22], 3, 4])
    qt_tree = QtNodeTreeModel(root)
    model_indexes = [qt_tree.nestedIndex(i) for i in sources]
    mime_data = qt_tree.mimeData(model_indexes)
    dest_mi = qt_tree.nestedIndex(dest)
    qt_tree.dropMimeData(
        mime_data,
        Qt.MoveAction,
        dest_mi.row(),
        dest_mi.column(),
        dest_mi.parent(),
    )
    expected = _recursive_make_group(expectation)
    _assert_models_synced(expected, qt_tree)


def test_qt_tree_model_deletion(qapp):
    """Test that we can delete items from a QTreeModel"""
    root = _recursive_make_group([0, 1, [20, [210, 211], 22], 3, 4])
    qt_tree = QtNodeTreeModel(root)
    _assert_models_synced(root, qt_tree)
    del root[2, 1]
    e = _recursive_make_group([0, 1, [20, 22], 3, 4])
    _assert_models_synced(e, qt_tree)


def test_qt_tree_model_insertion(qapp):
    """Test that we can append and insert items to a QTreeModel."""
    root = _recursive_make_group([0, 1, [20, [210, 211], 22], 3, 4])
    qt_tree = QtNodeTreeModel(root)
    _assert_models_synced(root, qt_tree)
    root[2, 1].append(Node(name='212'))
    e = _recursive_make_group([0, 1, [20, [210, 211, 212], 22], 3, 4])
    _assert_models_synced(e, qt_tree)

    root.insert(-2, Node(name='9'))
    e = _recursive_make_group([0, 1, [20, [210, 211, 212], 22], 9, 3, 4])
    _assert_models_synced(e, qt_tree)


def test_find_nodes(qapp):
    root = _recursive_make_group([0, 1, [20, [210, 211], 22], 3, 4])

    qt_tree = QtNodeTreeModel(root)
    _assert_models_synced(root, qt_tree)
    node = Node(name='212')
    root[2, 1].append(node)
    assert index_of(qt_tree, node).row() == 2
    assert not index_of(qt_tree, Node(name='new node')).isValid()


def test_node_tree_view(qtbot):
    root = _recursive_make_group([0, 1, [20, [210, 211], 22], 3, 4])
    root.selection.clear()
    assert not root.selection
    view = QtNodeTreeView(root)
    qmodel = view.model()
    qsel = view.selectionModel()
    qtbot.addWidget(view)

    # update selection in python
    root.selection.update([root[0], root[2, 0]])
    assert root[2, 0] in root.selection

    # check selection in Qt
    idx = {qmodel.getItem(i).index_from_root() for i in qsel.selectedIndexes()}
    assert idx == {(0,), (2, 0)}

    # clear selection in Qt
    qsel.clearSelection()
    # check selection in python
    assert not root.selection

    # update current in python
    root.selection._current = root[2, 1, 0]
    # check current in Qt
    assert root.selection._current == root[2, 1, 0]
    assert qmodel.getItem(qsel.currentIndex()).index_from_root() == (2, 1, 0)

    # clear current in Qt
    qsel.setCurrentIndex(QModelIndex(), qsel.Current)
    # check current in python
    assert root.selection._current is None


def test_flags(tree_model):
    """Some sanity checks on retrieving flags for nested items"""
    assert not tree_model.hasIndex(5, 0, tree_model.index(1))
    last = tree_model._root.pop()
    tree_model._root[1].append(last)
    assert tree_model.hasIndex(5, 0, tree_model.index(1))
    idx = tree_model.index(5, 0, tree_model.index(1))
    assert bool(tree_model.flags(idx) & Qt.ItemFlag.ItemIsEnabled)
