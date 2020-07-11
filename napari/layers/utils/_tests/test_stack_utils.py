from napari.layers.image.image import Image
from napari.layers.utils.stack_utils import stack_to_images, images_to_stack
import numpy as np


def test_stack_to_images():

    '''
    Test 2 channels, no colormap
    '''
    data = np.random.randint(0, 100, (10, 2, 128, 128))
    stack = Image(data)
    images = stack_to_images(stack, 1, colormap=None)

    assert isinstance(images, list)
    assert images[0].colormap[0] == 'magenta'
    assert len(images) == 2

    for i in images:
        assert type(stack) == type(i)
        assert i.data.shape == (10, 128, 128)

    '''
    Test 3 channels, last index, no colormap - RGB
    from rgb to single channels
    '''
    data = np.random.randint(0, 100, (10, 128, 128, 3))
    stack = Image(data)
    images = stack_to_images(stack, -1, colormap=None)

    assert isinstance(images, list)
    assert len(images) == 3

    for i in images:
        assert type(stack) == type(i)
        assert i.data.shape == (10, 128, 128)
        assert i.scale.shape == (3,)
        assert i.rgb is False

    '''
    Test 4 channels, first index, colormap
    '''
    data = np.random.randint(0, 100, (4, 128, 128))
    stack = Image(data)
    images = stack_to_images(stack, 0, colormap=['red', 'blue'])

    assert isinstance(images, list)
    assert len(images) == 4
    assert images[-2].colormap[0] == 'red'
    for i in images:
        assert type(stack) == type(i)
        assert i.data.shape == (128, 128)

    '''
    Split on axis 0 when rgb
    '''
    data = np.random.randint(0, 100, (10, 128, 128, 3))
    stack = Image(data)
    images = stack_to_images(stack, 0, colormap=None)

    assert isinstance(images, list)
    assert len(images) == 10
    for i in images:
        assert i.rgb
        assert type(stack) == type(i)
        assert i.data.shape == (128, 128, 3)

    '''
    Split when only one channel
    '''
    data = np.random.randint(0, 100, (10, 1, 128, 128))
    stack = Image(data)
    images = stack_to_images(stack, 1, colormap=['magma'])

    assert isinstance(images, list)
    assert len(images) == 1
    for i in images:
        assert i.rgb is False
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


if __name__ == '__main__':
    test_stack_to_images()
