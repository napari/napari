import pytest

from napari._qt.tree import QtNodeTreeModel
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
    out = []
    for item in lst:
        if isinstance(item, list):
            out.append(_recursive_make_group(item, level=level + 1))
        else:
            out.append(Node(name=str(item)))
    return Group(out, name=f'g{level}')


def test_tree_model(qtmodeltester):
    # https://pytest-qt.readthedocs.io/en/latest/modeltester.html
    root = _recursive_make_group([0, 1, [20, [210, 211], 22], 3, 4])
    model = QtNodeTreeModel(root)
    qtmodeltester.check(model)


def assert_models_synced(model: Group, qt_model: QtNodeTreeModel):
    for item in model.traverse():
        model_idx = qt_model.nestedIndex(item.index_from_root())
        node = qt_model.getItem(model_idx)
        assert item is node


def test_move_single_tree_item(tree_model):
    root = tree_model._root
    assert isinstance(root, Group)
    assert_models_synced(root, tree_model)

    root.move(0, 2)
    assert_models_synced(root, tree_model)

    root.move(3, 1)
    assert_models_synced(root, tree_model)


@pytest.mark.parametrize(
    'source, dest',
    [
        # indices           2       (2, 1)
        # original = [0, 1, [(2,0), [(2,1,0), (2,1,1)], (2,2)], 3, 4]
        [((2, 0), (2, 1, 1), (3,)), (-1)],
        [((2, 0), (2, 1, 1), (3,)), (1)],
        [((2, 1, 1),), (0,)],
        [((2, 1, 1),), ()],
    ],
)
def test_nested_move_multiple(source, dest):
    """Test that moving multiple indices works and emits right events."""

    root = _recursive_make_group([0, 1, [20, [210, 211], 22], 3, 4])
    qt_tree = QtNodeTreeModel(root)
    root.move_multiple(source, dest)
    assert_models_synced(root, qt_tree)
