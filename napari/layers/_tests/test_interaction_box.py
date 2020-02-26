import numpy as np

from ..interaction_box import InteractionBox


def test_interaction_box_initialization():
    """Test the creation of the interaction box."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    interaction_box = InteractionBox(points=data)
    assert len(interaction_box._box) == 10

    interaction_box = InteractionBox(points=data, show=True)
    (
        vertices,
        face_color,
        edge_color,
        pos,
        width,
    ) = interaction_box._compute_vertices_and_box()
    assert len(vertices) == 9
    assert len(pos) == 7

    interaction_box = InteractionBox(points=data, show=True, show_handle=False)
    (
        vertices,
        face_color,
        edge_color,
        pos,
        width,
    ) = interaction_box._compute_vertices_and_box()
    assert len(vertices) == 4
    assert len(pos) == 5

    interaction_box = InteractionBox(points=data, show=False)
    (
        vertices,
        face_color,
        edge_color,
        pos,
        width,
    ) = interaction_box._compute_vertices_and_box()
    assert len(vertices) == 0
    assert pos is None

    interaction_box = InteractionBox(points=[], show=True)
    assert interaction_box._box is None


def test_interaction_box_update():
    """Test the creation of the interaction box."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    interaction_box = InteractionBox()
    (
        vertices,
        face_color,
        edge_color,
        pos,
        width,
    ) = interaction_box._compute_vertices_and_box()
    assert len(vertices) == 0
    assert pos is None

    interaction_box.points = data
    assert len(interaction_box._box) == 10
    (
        vertices,
        face_color,
        edge_color,
        pos,
        width,
    ) = interaction_box._compute_vertices_and_box()
    assert len(vertices) == 0
    assert pos is None

    interaction_box.show = True
    (
        vertices,
        face_color,
        edge_color,
        pos,
        width,
    ) = interaction_box._compute_vertices_and_box()
    assert len(vertices) == 9
    assert len(pos) == 7

    interaction_box.show = False
    (
        vertices,
        face_color,
        edge_color,
        pos,
        width,
    ) = interaction_box._compute_vertices_and_box()
    assert len(vertices) == 0
    assert pos is None
