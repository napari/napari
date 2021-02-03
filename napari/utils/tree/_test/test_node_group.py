from textwrap import dedent

import pytest

from napari.utils.tree import Group, Node


@pytest.fixture
def tree():
    root = Group(
        [
            Node(name="6"),
            Group(
                [
                    Node(name="1"),
                    Group([Node(name="2"), Node(name="3")], name="g2"),
                    Node(name="4"),
                    Node(name="5"),
                    Node(name="tip"),
                ],
                name="g1",
            ),
            Node(name="7"),
            Node(name="8"),
        ],
        name="root",
    )
    return root


def test_tree_str(tree):
    expected = dedent(
        """
        root
          ├──6
          ├──g1
          │  ├──1
          │  ├──g2
          │  │  ├──2
          │  │  └──3
          │  ├──4
          │  ├──5
          │  └──tip
          ├──7
          └──8"""
    ).strip()
    assert str(tree) == expected


def test_node_indexing(tree):
    """Test that nodes know their index relative to parent and root."""
    root: Group = tree
    assert root.is_group()
    assert not root[0].is_group()

    assert root.index_from_root() == ()
    assert root.index_in_parent() == 0
    g1 = root[1]
    assert g1.index_from_root() == (1,)
    assert g1.index_in_parent() == 1
    g11 = g1[1]
    assert g11.index_from_root() == (1, 1)
    assert g11.index_in_parent() == 1
    g110 = g11[0]
    assert g110.index_from_root() == (1, 1, 0)
    assert g110.index_in_parent() == 0
    assert g110.name == '2'

    g110.emancipate()
    assert g110.index_from_root() == ()
    assert g110.index_in_parent() == 0

    with pytest.raises(IndexError) as e:
        g110.emancipate()
    assert "Cannot emancipate orphaned Node" in str(e)


def test_traverse(tree):
    """Test depth first traversal."""
    names = [x.name for x in tree.traverse()]
    e = ['root', '6', 'g1', '1', 'g2', '2', '3', '4', '5', 'tip', '7', '8']
    assert names == e
    assert tree.is_group()
    g1 = tree[1]
    assert g1.parent is tree
    assert g1.name == 'g1' and g1.is_group()
    g2 = g1[1]
    assert g2.parent is g1
    assert g2.name == 'g2' and g2.is_group()


def test_deletion(tree):
    """Test that deletion removes parent"""
    g1 = tree[1]
    n1 = g1[0]
    assert g1.parent is tree
    del tree[1]
    assert g1.parent is not tree
    assert [x.name for x in tree.traverse()] == ['root', '6', '7', '8']

    assert n1.parent is g1  # the g1 tree is still intact

    expected = ['g1', '1', 'g2', '2', '3', '4', '5', 'tip']
    assert [x.name for x in g1.traverse()] == expected
    # we can also delete slices, including extended slices
    del g1[1::2]
    assert [x.name for x in g1.traverse()] == ['g1', '1', '4', 'tip']
