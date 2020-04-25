import pytest
from napari.components.add_layers_mixin import prune_kwargs

TEST_KWARGS = {
    'scale': (0.75, 1),
    'blending': 'translucent',
    'num_colors': 10,
    'edge_color': 'red',
    'z_index': 20,
    'edge_width': 2,
    'face_color': 'white',
    'is_multiscale': False,
    'name': 'name',
    'extra_kwarg': 'never_included',
}

EXPECTATIONS = [
    (
        'image',
        {
            'scale': (0.75, 1),
            'blending': 'translucent',
            'is_multiscale': False,
            'name': 'name',
        },
    ),
    (
        'labels',
        {
            'scale': (0.75, 1),
            'num_colors': 10,
            'is_multiscale': False,
            'name': 'name',
            'blending': 'translucent',
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
            'name': 'name',
        },
    ),
    (
        'shapes',
        {
            'scale': (0.75, 1),
            'edge_color': 'red',
            'z_index': 20,
            'edge_width': 2,
            'face_color': 'white',
            'name': 'name',
            'blending': 'translucent',
        },
    ),
    (
        'vectors',
        {
            'scale': (0.75, 1),
            'edge_color': 'red',
            'edge_width': 2,
            'name': 'name',
            'blending': 'translucent',
        },
    ),
    (
        'surface',
        {'blending': 'translucent', 'scale': (0.75, 1), 'name': 'name'},
    ),
    ('layer', {}),
]

ids = [i[0] for i in EXPECTATIONS]


@pytest.mark.parametrize('label_type, expectation', EXPECTATIONS, ids=ids)
def test_prune_kwargs(label_type, expectation):
    assert prune_kwargs(TEST_KWARGS, label_type) == expectation


def test_prune_kwargs_raises():
    with pytest.raises(ValueError):
        prune_kwargs({}, 'nonexistent_layer_type')
