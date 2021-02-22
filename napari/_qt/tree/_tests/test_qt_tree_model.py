import pytest
from qtpy.QtCore import Qt

from napari._qt.tree import QtNodeTreeModel, QtNodeTreeView
from napari.utils.events._tests.test_evented_list import POS_INDICES
from napari.utils.tree import Group, Node


@pytest.fixture
def tree_model():
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


@pytest.mark.parametrize('sources, dest, expectation', POS_INDICES)
def test_nested_move_multiple(sources, dest, expectation):
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


def test_qt_tree_model_deletion():
    """Test that we can delete items from a QTreeModel"""
    root = _recursive_make_group([0, 1, [20, [210, 211], 22], 3, 4])
    qt_tree = QtNodeTreeModel(root)
    _assert_models_synced(root, qt_tree)
    del root[2, 1]
    e = _recursive_make_group([0, 1, [20, 22], 3, 4])
    _assert_models_synced(e, qt_tree)


def test_qt_tree_model_insertion():
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


def test_find_nodes():
    root = _recursive_make_group([0, 1, [20, [210, 211], 22], 3, 4])

    qt_tree = QtNodeTreeModel(root)
    _assert_models_synced(root, qt_tree)
    node = Node(name='212')
    root[2, 1].append(node)
    assert qt_tree.findIndex(node).row() == 2

    with pytest.raises(IndexError):
        qt_tree.findIndex(Node(name='new node'))


def test_view_smoketest(qtbot):
    root = _recursive_make_group([0, 1, [20, [210, 211], 22], 3, 4])
    view = QtNodeTreeView(root)
    qtbot.addWidget(view)
