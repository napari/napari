import numpy as np

from napari.components.interaction_box import InteractionBox
from napari.utils.transforms import Affine
from napari.viewer import Viewer


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


def test_interaction_box_changes_with_layer_transform():
    viewer = Viewer(show=False)
    image_layer = viewer.add_image(np.random.random((28, 28)))
    image_layer.mode = 'transform'
    initial_selection_box = np.copy(
        viewer.overlays.interaction_box.transform.affine_matrix
    )
    viewer.layers[0].scale = [5, 5]
    assert not np.allclose(
        initial_selection_box,
        viewer.overlays.interaction_box.transform.affine_matrix,
    )
