import numpy as np
import pytest

from napari.layers import LayerGroup, Image


@pytest.fixture
def layergroup_tree():
    """Build a layergroup to use for testing.

    tree
    +--branch1
    |  +--image1_on_branch1
    |  +--image2_on_branch1
    +--branch2
    |  +--image1_on_branch2
    +--branch3
    |  +--branch4
    |  |  +--image1_on_branch4_on_branch3
    |  |  +--image2_on_branch4_on_branch3
    |  +--branch5
    |  |  +--image1_on_branch5_on_branch3
    |  +--image1_on_branch3
    +--single_img

    """

    data = np.random.random((100, 100))

    tree = LayerGroup(name='tree')

    branch1 = LayerGroup(name='branch1')
    branch1.append(Image(data, name='image1_on_branch1'))
    branch1.append(Image(data, name='image2_on_branch1'))

    branch2 = LayerGroup(name='branch2')
    branch2.append(Image(data, name='image1_on_branch2'))

    branch3 = LayerGroup(name='branch3')
    branch4 = LayerGroup(name='branch4')
    branch5 = LayerGroup(name='branch5')
    branch4.append(Image(data, name='image1_on_branch4_on_branch3'))
    branch4.append(Image(data, name='image2_on_branch4_on_branch3'))
    branch5.append(Image(data, name='image1_on_branch5_on_branch3'))
    branch3.append(branch4)
    branch3.append(branch5)
    branch3.append(Image(data, name='image1_on_branch3'))

    tree.append(branch1)
    tree.append(branch2)
    tree.append(branch3)
    tree.append(Image(data, name='single_img'))

    return tree


def test_construct_layergroup(layergroup_tree):
    assert len(layergroup_tree) == 8  # number of non-group layers contained
    expected_layergroup_string = (
        "tree\n"
        "  +--branch1\n"
        "  |  +--image1_on_branch1\n"
        "  |  +--image2_on_branch1\n"
        "  +--branch2\n"
        "  |  +--image1_on_branch2\n"
        "  +--branch3\n"
        "  |  +--branch4\n"
        "  |  |  +--image1_on_branch4_on_branch3\n"
        "  |  |  +--image2_on_branch4_on_branch3\n"
        "  |  +--branch5\n"
        "  |  |  +--image1_on_branch5_on_branch3\n"
        "  |  +--image1_on_branch3\n"
        "  +--single_img"
    )
    assert str(layergroup_tree) == expected_layergroup_string
