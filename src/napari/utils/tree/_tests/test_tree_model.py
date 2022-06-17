from textwrap import dedent

import pytest

from napari.utils.tree import Group, Node


@pytest.fixture
def tree():
    return Group(
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


def test_tree_str(tree):
    expected = dedent(
        """
        root
          ├──1
          ├──g1
          │  ├──2
          │  ├──g2
          │  │  ├──3
          │  │  └──4
          │  ├──5
          │  ├──6
          │  └──7
          ├──8
          └──9"""
    ).strip()
    assert str(tree) == expected


def test_node_indexing(tree: Group):
    expected_indices = [
        0,
        1,
        (1, 0),
        (1, 1),
        (1, 1, 0),
        (1, 1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        2,
        3,
    ]
    assert list(tree._iter_indices()) == expected_indices

    for index in tree._iter_indices():
        assert tree.index(tree[index]) == index

        item = tree[index]
        if item.parent:
            assert item.parent.index(item) is not None


def test_relative_node_indexing(tree):
    """Test that nodes know their index relative to parent and root."""
    root: Group[Node] = tree
    assert root.is_group()
    assert not root[0].is_group()

    assert root.index_from_root() == ()
    assert root.index_in_parent() is None
    g1 = root[1]
    assert g1.name == 'g1'
    assert g1.index_in_parent() == 1
    assert g1.index_from_root() == (1,)
    g1_1 = g1[1]
    assert g1_1.name == 'g2'
    assert g1_1.parent is g1
    assert g1_1.parent.parent is root
    assert g1_1 is tree[1, 1]  # nested index variant

    assert g1_1.index_from_root() == (1, 1)
    assert g1_1.index_in_parent() == 1
    g1_1_0 = g1_1[0]
    assert g1_1_0.index_from_root() == (1, 1, 0)
    assert g1_1_0.index_in_parent() == 0
    assert g1_1_0.name == '3'
    assert g1_1_0 is tree[1, 1, 0]  # nested index variant

    g1_1_0.unparent()
    assert g1_1_0.index_from_root() == ()
    assert g1_1_0.index_in_parent() is None

    with pytest.raises(IndexError) as e:
        g1_1_0.unparent()
    assert "Cannot unparent orphaned Node" in str(e)


def test_traverse(tree):
    """Test depth first traversal."""
    # iterating a group just returns its children
    assert [x.name for x in tree] == ['1', 'g1', '8', '9']
    # traversing a group does a depth first traversal, including both groups
    # and nodes
    names = [x.name for x in tree.traverse()]
    e = ['root', '1', 'g1', '2', 'g2', '3', '4', '5', '6', '7', '8', '9']
    assert names == e

    # traversing leaves_only=True returns only the Nodes, not the Groups
    names = [x.name for x in tree.traverse(leaves_only=True)]
    e = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    assert names == e

    assert tree.is_group()
    g1 = tree[1]
    assert g1.parent is tree
    assert g1.name == 'g1' and g1.is_group()
    g2 = g1[1]
    assert g2.parent is g1
    assert g2.name == 'g2' and g2.is_group()


def test_slicing(tree):
    """Indexing into a group returns a group instance."""
    assert tree.is_group()
    slc = tree[::-2]  # take every other item, starting from the end
    assert [x.name for x in slc] == ['9', 'g1']
    assert slc.is_group()
    expected = ['Group', '9', 'g1', '2', 'g2', '3', '4', '5', '6', '7']
    assert [x.name for x in slc.traverse()] == expected


def test_contains(tree):
    """Test that the ``in`` operator works for nested nodes."""
    g1 = tree[1]
    assert g1.name == 'g1'
    assert g1 in tree

    g1_0 = g1[0]
    assert g1_0.name == '2'
    assert g1_0 in g1
    assert g1_0 in tree

    # If you need to know if an item is an immediate child, you can use parent
    assert g1.parent is tree
    assert g1_0.parent is g1

    g2 = g1[1]
    assert g2.name == 'g2'
    assert g2.is_group()
    assert g2 in tree

    g2_0 = g2[0]
    assert g2_0.name == '3'


def test_deletion(tree):
    """Test that deletion removes parent"""
    g1 = tree[1]  # first group in tree
    assert g1.parent is tree
    assert g1 in tree
    n1 = g1[0]  # first item in group1

    del tree[1]  # delete g1 from the tree
    assert g1.parent is not tree
    assert g1 not in tree
    # the tree no longer has g1 or any of its children
    assert [x.name for x in tree.traverse()] == ['root', '1', '8', '9']

    # g1 remains intact
    expected = ['g1', '2', 'g2', '3', '4', '5', '6', '7']
    assert [x.name for x in g1.traverse()] == expected
    expected = ['2', '3', '4', '5', '6', '7']
    assert [x.name for x in g1.traverse(leaves_only=True)] == expected

    # we can also delete slices, including extended slices
    del g1[1::2]
    assert n1.parent is g1  # the g1 tree is still intact
    assert [x.name for x in g1.traverse()] == ['g1', '2', '5', '7']


def test_nested_deletion(tree):
    """Test that we can delete nested indices from the root."""
    # a tree is a NestedEventedList, so we can use nested_indices
    node5 = tree[1, 2]
    assert node5.name == '5'
    del tree[1, 2]
    assert node5 not in tree

    # nested indices may also be slices
    g2 = tree[1, 1]
    node4 = g2[1]
    assert node4 in tree
    del tree[1, 1, :]  # delete all members of g2 inside of tree
    assert node4 not in tree  # node4 is gone
    assert g2 == []
    assert g2 in tree  # the group itself remains in the tree


def test_deep_index(tree: Group):
    """Test deep indexing"""

    node = tree[(1, 0)]
    assert tree.index(node) == (1, 0)


def test_remove_selected(tree: Group):
    """Test remove_selected works, with nested"""
    node = tree[(1, 0)]
    tree.selection.active = node
    tree.remove_selected()


def test_nested_custom_lookup(tree: Group):
    tree._lookup = {str: lambda x: x.name}

    # first level
    g1 = tree[1]
    assert g1.name == 'g1'  # index with integer as usual
    assert tree.index("g1") == 1
    assert tree['g1'] == g1  # index with string also works

    # second level
    g1_2 = g1[2]
    assert tree[1, 2].name == '5'
    assert tree.index('5') == (1, 2)
    assert tree['5'] == g1_2
