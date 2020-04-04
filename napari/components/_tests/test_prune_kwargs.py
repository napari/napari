import pytest
from napari.components.add_layers_mixin import prune_kwargs

TEST_KWARGS = {
    'scale': (0.75, 1),
    'vectors_scale': (1, 1, 1),
    'image_blending': 'additive',
    'points_blending': 'translucent',
    'num_colors': 10,
    'edge_color': 'red',
    'z_index': 20,
    'edge_width': 2,
    'shapes_edge_width': 10,
    'face_color': 'white',
    'is_pyramid': False,
    'name': 'name',
    'image_name': 'image',
    'points_name': 'points',
    'extra_kwarg': 'never_included',
}

EXPECTATIONS = [
    (
        'image',
        {
            'scale': (0.75, 1),
            'blending': 'additive',
            'is_pyramid': False,
            'name': 'image',
        },
    ),
    (
        'labels',
        {
            'scale': (0.75, 1),
            'num_colors': 10,
            'is_pyramid': False,
            'name': 'name',
        },
    ),
    (
        'points',
        {
            'scale': (0.75, 1),
            'blending': 'translucent',
            'edge_color': 'red',
            'edge_width': 2,
            'face_color': 'white',
            'name': 'points',
        },
    ),
    (
        'shapes',
        {
            'scale': (0.75, 1),
            'edge_color': 'red',
            'z_index': 20,
            'edge_width': 10,
            'face_color': 'white',
            'name': 'name',
        },
    ),
    (
        'vectors',
        {
            'scale': (1, 1, 1),
            'edge_color': 'red',
            'edge_width': 2,
            'name': 'name',
        },
    ),
    ('surface', {'scale': (0.75, 1), 'name': 'name'}),
    ('layer', {}),
    ('path', {}),
]

ids = [i[0] for i in EXPECTATIONS]


@pytest.mark.parametrize('label_type, expectation', EXPECTATIONS, ids=ids)
def test_prune_kwargs(label_type, expectation):
    assert prune_kwargs(TEST_KWARGS, label_type) == expectation


def test_prune_kwargs_raises():
    with pytest.raises(ValueError):
        prune_kwargs({}, 'nonexistent_layer_type')
