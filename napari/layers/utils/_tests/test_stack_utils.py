from napari.layers.image.image import Image
from napari.layers.utils.stack_utils import stack_to_images, images_to_stack
import numpy as np


def test_stack_to_images():

    data = np.random.randint(0, 100, (10, 3, 128, 128))
    stack = Image(data)
    images = stack_to_images(stack, 1, colormap=['magenta', 'gray', 'blue'])

    assert isinstance(images, list)
    assert len(images) == 3

    for i in images:
        assert type(stack) == type(i)
        assert i.data.shape == (10, 128, 128)


def test_images_to_stack():

    images = [
        Image(np.random.randint(0, 255, (10, 128, 128))) for _ in range(3)
    ]

    stack = images_to_stack(images, 1, colormap='green', scale=(3, 1, 1, 1))

    assert isinstance(stack, Image)
    assert stack.data.shape == (10, 3, 128, 128)
    assert stack.colormap[0] == 'green'
    assert list(stack.scale) == [3, 1, 1, 1]
