from napari.layers import Image
from napari.layers.utils.stack_utils import StackUtils
import numpy as np

s = StackUtils()


def test_stack_to_images_basic():

    '''Test the a 2 channel zcyx stack is split into 2 image layers'''
    data = np.random.randint(0, 100, (10, 2, 128, 128))
    stack = Image(data)
    images = s.stack_to_images(stack, 1, colormap=None)

    assert isinstance(images, list)
    assert images[0].colormap[0] == 'magenta'
    assert len(images) == 2

    for i in images:
        assert type(stack) == type(i)
        assert i.data.shape == (10, 128, 128)


def test_stack_to_images_rgb():
    '''Test 3 channel RGB image (channel axis = -1) into single channels.'''
    data = np.random.randint(0, 100, (10, 128, 128, 3))
    stack = Image(data)
    images = s.stack_to_images(stack, -1, colormap=None)

    assert isinstance(images, list)
    assert len(images) == 3

    for i in images:
        assert type(stack) == type(i)
        assert i.data.shape == (10, 128, 128)
        assert i.scale.shape == (3,)
        assert i.rgb is False


def test_stack_to_images_4_channels():
    '''Test 4 channel stack with channel as the first index is split
    into mutliple channels and colormap keyword'''
    data = np.random.randint(0, 100, (4, 128, 128))
    stack = Image(data)
    images = s.stack_to_images(stack, 0, colormap=['red', 'blue'])

    assert isinstance(images, list)
    assert len(images) == 4
    assert images[-2].colormap[0] == 'red'
    for i in images:
        assert type(stack) == type(i)
        assert i.data.shape == (128, 128)


def test_stack_to_images_0_rgb():
    '''Split RGB along the first axis (z or t) so the images remain rgb'''
    data = np.random.randint(0, 100, (10, 128, 128, 3))
    stack = Image(data)
    images = s.stack_to_images(stack, 0, colormap=None)

    assert isinstance(images, list)
    assert len(images) == 10
    for i in images:
        assert i.rgb
        assert type(stack) == type(i)
        assert i.data.shape == (128, 128, 3)


def test_stack_to_images_1_channel():
    '''Split when only one channel'''
    data = np.random.randint(0, 100, (10, 1, 128, 128))
    stack = Image(data)
    images = s.stack_to_images(stack, 1, colormap=['magma'])

    assert isinstance(images, list)
    assert len(images) == 1
    for i in images:
        assert i.rgb is False
        assert type(stack) == type(i)
        assert i.data.shape == (10, 128, 128)


def test_images_to_stack_with_scale():
    '''Test that input images are combined to a stack with scale and translate.'''
    images = [
        Image(np.random.randint(0, 255, (10, 128, 128))) for _ in range(3)
    ]

    stack = s.images_to_stack(
        images, 1, colormap='green', scale=(3, 1, 1, 1), translate=(1, 0, 2, 3)
    )

    assert isinstance(stack, Image)
    assert stack.data.shape == (10, 3, 128, 128)
    assert stack.colormap[0] == 'green'
    assert list(stack.scale) == [3, 1, 1, 1]
    assert list(stack.translate) == [1, 0, 2, 3]


def test_images_to_stack_none_scale():
    '''Test combining images using scale and translate from first image in list'''
    images = [
        Image(
            np.random.randint(0, 255, (10, 128, 128)),
            scale=(4, 1, 1),
            translate=(0, -1, 2),
        )
        for _ in range(3)
    ]

    stack = s.images_to_stack(images, 1, colormap='green')

    assert isinstance(stack, Image)
    assert stack.data.shape == (10, 3, 128, 128)
    assert stack.colormap[0] == 'green'
    assert list(stack.scale) == [4, 1, 1, 1]
    assert list(stack.translate) == [0, 0, -1, 2]
