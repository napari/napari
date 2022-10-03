import numpy as np

from napari.components.overlays.interaction_box import InteractionBox
from napari.utils.transforms import Affine


def test_creation():
    """Test creating interaction box object"""
    interaction_box = InteractionBox()
    assert interaction_box is not None


def test_box_from_points():
    """Test whether setting points creates a axis-aligned containing box"""
    interaction_box = InteractionBox()
    interaction_box.points = np.array([[1, 0], [3, 2], [-1, 1]])
    resulting_box = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [3.0, 2.0],
            [1.0, 2.0],
            [-1.0, 2.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    np.testing.assert_equal(interaction_box._box, resulting_box)


def test_transform():
    """Tests whether setting a transform changes the box adequatly"""
    interaction_box = InteractionBox()
    interaction_box.points = np.array([[1, 0], [3, 2], [-1, 1]])
    resulting_box = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [3.0, 2.0],
            [1.0, 2.0],
            [-1.0, 2.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    interaction_box.transform = Affine(rotate=45)
    resulting_box = Affine(rotate=45)(resulting_box)
    np.testing.assert_equal(interaction_box._box, resulting_box)
