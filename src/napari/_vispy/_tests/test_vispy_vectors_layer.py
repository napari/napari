import pytest


@pytest.mark.parametrize(
    ('initial_vector_style', 'new_vector_style'),
    [
        ('line', 'line'),
        ('line', 'triangle'),
        ('line', 'arrow'),
        ('triangle', 'line'),
        ('triangle', 'triangle'),
        ('triangle', 'arrow'),
        ('arrow', 'line'),
        ('arrow', 'triangle'),
        ('arrow', 'arrow'),
    ],
)
def test_vector_style_change(
    make_napari_viewer, initial_vector_style, new_vector_style
):
    viewer = make_napari_viewer()
    vector_layer = viewer.add_vectors(
        vector_style=initial_vector_style, name='vectors'
    )

    class Counter:
        def __init__(self):
            self.count = 0

        def increment_count(self, event):
            self.count += 1

    counter = Counter()
    vector_layer.events.vector_style.connect(counter.increment_count)

    vector_layer.vector_style = new_vector_style

    if initial_vector_style == new_vector_style:
        assert counter.count == 0
    else:
        assert counter.count == 1
