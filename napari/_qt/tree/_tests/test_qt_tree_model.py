import pytest

from napari._qt.tree import QtNodeTreeModel
from napari.utils.events._tests.test_evented_list import MULTIMOVE_INDICES
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
        assert item is node


def test_tree_model(qtmodeltester):
    """Basic tests on the qtabstractitem model implementation.

    https://pytest-qt.readthedocs.io/en/latest/modeltester.html
    """
    root = _recursive_make_group([0, 1, [20, [210, 211], 22], 3, 4])
    model = QtNodeTreeModel(root)
    qtmodeltester.check(model)


def test_move_single_tree_item(tree_model):
    """Test moving a single item."""
    root = tree_model._root
    assert isinstance(root, Group)
    _assert_models_synced(root, tree_model)
    root.move(0, 2)
    _assert_models_synced(root, tree_model)
    root.move(3, 1)
    _assert_models_synced(root, tree_model)


@pytest.mark.parametrize('source, dest, _', MULTIMOVE_INDICES)
def test_nested_move_multiple(source, dest, _):
    """Test that models stay in sync with complicated moves.

    It's possible that this is a completely repetive test, since there should
    only be "one source of truth here" anyway (the python model).
    """
    root = _recursive_make_group([0, 1, [20, [210, 211], 22], 3, 4])
    qt_tree = QtNodeTreeModel(root)
    root.move_multiple(source, dest)
    _assert_models_synced(root, qt_tree)
