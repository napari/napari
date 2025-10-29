import dask.array as da
import numpy as np
import pytest
import zarr

from napari.layers import Image
from napari.layers.base._base_constants import Blending
from napari.layers.utils.stack_utils import (
    images_to_stack,
    merge_rgb,
    slice_from_axis,
    split_channels,
    split_rgb,
    stack_to_images,
)
from napari.utils.transforms import Affine


def test_stack_to_images_basic():
    """Test that a 2 channel zcyx stack is split into 2 image layers"""
    data = np.random.randint(0, 100, (10, 2, 128, 128))
    stack = Image(data)
    images = stack_to_images(stack, 1, colormap=None)

    assert isinstance(images, list)
    assert images[0].colormap.name == 'magenta'
    assert len(images) == 2

    for i in images:
        assert type(stack) is type(i)
        assert i.data.shape == (10, 128, 128)


def test_stack_to_images_multiscale():
    """Test that a 3 channel multiscale image returns 3 multiscale images."""
    data = []
    data.append(np.zeros((3, 128, 128)))
    data.append(np.zeros((3, 64, 64)))
    data.append(np.zeros((3, 32, 32)))
    data.append(np.zeros((3, 16, 16)))

    stack = Image(data)
    images = stack_to_images(stack, 0)

    assert len(images) == 3
    assert len(images[0].data) == 4
    assert images[0].data[-1].shape[-1] == 16
    assert images[1].data[-1].shape[-1] == 16
    assert images[2].data[-1].shape[-1] == 16


def test_stack_to_images_rgb():
    """Test 3 channel RGB image (channel axis = -1) into single channels."""
    data = np.random.randint(0, 100, (10, 128, 128, 3))
    stack = Image(data)
    images = stack_to_images(stack, -1, colormap=None)

    assert isinstance(images, list)
    assert len(images) == 3

    for i in images:
        assert type(stack) is type(i)
        assert i.data.shape == (10, 128, 128)
        assert i.scale.shape == (3,)
        assert i.rgb is False


def test_stack_to_images_4_channels():
    """Test 4x128x128 stack is split into 4 channels w/ colormap keyword"""
    data = np.random.randint(0, 100, (4, 128, 128))
    stack = Image(data)
    images = stack_to_images(stack, 0, colormap=['red', 'blue'])

    assert isinstance(images, list)
    assert len(images) == 4
    assert images[-2].colormap.name == 'red'
    for i in images:
        assert type(stack) is type(i)
        assert i.data.shape == (128, 128)


def test_stack_to_images_0_rgb():
    """Split RGB along the first axis (z or t) so the images remain rgb"""
    data = np.random.randint(0, 100, (10, 128, 128, 3))
    stack = Image(data)
    images = stack_to_images(stack, 0, colormap=None)

    assert isinstance(images, list)
    assert len(images) == 10
    for i in images:
        assert i.rgb
        assert type(stack) is type(i)
        assert i.data.shape == (128, 128, 3)


def test_stack_to_images_1_channel():
    """Split when only one channel"""
    data = np.random.randint(0, 100, (10, 1, 128, 128))
    stack = Image(data)
    images = stack_to_images(stack, 1, colormap=['magma'])

    assert isinstance(images, list)
    assert len(images) == 1
    for i in images:
        assert i.rgb is False
        assert type(stack) is type(i)
        assert i.data.shape == (10, 128, 128)


def test_images_to_stack_with_scale():
    """Test that 3-Image list is combined to stack with scale and translate."""
    images = [
        Image(np.random.randint(0, 255, (10, 128, 128))) for _ in range(3)
    ]

    stack = images_to_stack(
        images, 1, colormap='green', scale=(3, 1, 1, 1), translate=(1, 0, 2, 3)
    )

    assert isinstance(stack, Image)
    assert stack.data.shape == (10, 3, 128, 128)
    assert stack.colormap.name == 'green'
    assert list(stack.scale) == [3, 1, 1, 1]
    assert list(stack.translate) == [1, 0, 2, 3]


def test_images_to_stack_none_scale():
    """Test combining images using scale & translate from 1st image in list"""
    images = [
        Image(
            np.random.randint(0, 255, (10, 128, 128)),
            scale=(4, 1, 1),
            translate=(0, -1, 2),
        )
        for _ in range(3)
    ]

    stack = images_to_stack(images, 1, colormap='green')

    assert isinstance(stack, Image)
    assert stack.data.shape == (10, 3, 128, 128)
    assert stack.colormap.name == 'green'
    assert list(stack.scale) == [4, 1, 1, 1]
    assert list(stack.translate) == [0, 0, -1, 2]


def test_images_to_stack_multiscale():
    data = np.zeros((100, 100))
    images = [
        Image([data, data[::2, ::2]]),
        Image([data, data[::2, ::2]]),
    ]
    assert images[0].multiscale is True
    assert images[1].multiscale is True

    stack = images_to_stack(images, 0, colormap='green')

    assert isinstance(stack, Image)
    assert stack.multiscale is True
    assert stack.data.shape == (2, 100, 100)


def test_split_and_merge_rgb():
    """Test merging 3 images with RGB colormaps into single RGB image."""
    # Make an RGB
    data = np.random.randint(0, 100, (10, 128, 128, 3))
    stack = Image(data)
    assert stack.rgb is True

    # split the RGB into 3 images
    images = split_rgb(stack)
    assert len(images) == 3
    colormaps = {image.colormap.name for image in images}
    assert colormaps == {'red', 'green', 'blue'}

    # merge the 3 images back into an RGB
    rgb_image = merge_rgb(images)
    assert rgb_image.rgb is True


def test_split_and_merge_rgba():
    """Test splitting a rgba image into channels and then re-merging."""
    data = np.random.randint(0, 100, (10, 128, 128, 4))
    # set channels to distinct values to aid in confirming re-merging image
    data[0] = 1
    data[1] = 2
    data[2] = 3
    data[3] = 4
    stack = Image(data)
    assert stack.rgb is True

    # split the rgb into 4 images
    images = split_rgb(stack, with_alpha=True)
    assert len(images) == 4
    colormaps = {image.colormap.name for image in images}
    # gray should be assigned to alpha channel
    assert colormaps == {'red', 'green', 'blue', 'gray'}

    # merge the 4 images back into a rgba image
    rgb_image = merge_rgb(images)
    assert rgb_image.rgb is True
    assert rgb_image.data.shape[-1] == 4
    # confirm that channel are assigned correctly
    assert (rgb_image.data[0] == 1).all()
    assert (rgb_image.data[1] == 2).all()
    assert (rgb_image.data[2] == 3).all()
    assert (rgb_image.data[3] == 4).all()


@pytest.mark.parametrize(
    'stack_blending', [blending.value for blending in Blending]
)
def test_split_rgb_blending(stack_blending):
    """Test blending settings on a split RGB image."""
    # Make an RGB
    data = np.random.randint(0, 100, (10, 128, 128, 3))
    stack = Image(data)
    stack.blending = stack_blending

    # split the RGB into 3 images
    images = split_rgb(stack)
    blendings = [image.blending for image in images]
    assert blendings == [stack_blending, 'additive', 'additive']


@pytest.mark.parametrize(
    'stack_blending', [blending.value for blending in Blending]
)
def test_split_rgba_blending(stack_blending):
    """Test blending settings on a split RGBA image."""
    # Make an RGBA
    data = np.random.randint(0, 100, (10, 128, 128, 4))
    stack = Image(data)
    stack.blending = stack_blending

    # split the rgb into 4 images
    images = split_rgb(stack, with_alpha=True)
    blendings = [image.blending for image in images]
    # multiplicative should be assigned to alpha channel
    assert blendings == [
        stack_blending,
        'additive',
        'additive',
        'multiplicative',
    ]


@pytest.fixture(
    params=[
        {
            'rgb': None,
            'colormap': None,
            'contrast_limits': None,
            'gamma': 1,
            'interpolation': 'nearest',
            'rendering': 'mip',
            'iso_threshold': 0.5,
            'attenuation': 0.5,
            'name': None,
            'metadata': None,
            'scale': None,
            'translate': None,
            'opacity': 1,
            'blending': None,
            'visible': True,
            'multiscale': None,
            'rotate': None,
            'affine': None,
        },
        {
            'rgb': None,
            'colormap': None,
            'rendering': 'mip',
            'attenuation': 0.5,
            'metadata': None,
            'scale': None,
            'opacity': 1,
            'visible': True,
            'multiscale': None,
        },
        {},
    ],
    ids=['full-kwargs', 'partial-kwargs', 'empty-kwargs'],
)
def kwargs(request):
    return dict(request.param)


def test_split_channels(kwargs):
    """Test split_channels with shape (3,128,128) expecting 3 (128,128)"""
    data = np.zeros((3, 128, 128))
    result_list = split_channels(data, 0, **kwargs)

    assert len(result_list) == 3
    for d, _meta, _ in result_list:
        assert d.shape == (128, 128)


def test_split_channels_multiscale(kwargs):
    """Test split_channels with multiscale expecting List[LayerData]"""
    data = []
    data.append(np.zeros((3, 128, 128)))
    data.append(np.zeros((3, 64, 64)))
    data.append(np.zeros((3, 32, 32)))
    data.append(np.zeros((3, 16, 16)))
    result_list = split_channels(data, 0, **kwargs)

    assert len(result_list) == 3
    for ds, m, _ in result_list:
        assert m['multiscale'] is True
        assert ds[0].shape == (128, 128)
        assert ds[1].shape == (64, 64)
        assert ds[2].shape == (32, 32)
        assert ds[3].shape == (16, 16)


def test_split_channels_blending(kwargs):
    """Test split_channels with shape (3,128,128) expecting 3 (128,128)"""
    kwargs['blending'] = 'translucent'
    data = np.zeros((3, 128, 128))
    result_list = split_channels(data, 0, **kwargs)

    assert len(result_list) == 3
    for d, meta, _ in result_list:
        assert d.shape == (128, 128)
        assert meta['blending'] == 'translucent'


def test_split_channels_missing_keywords():
    data = np.zeros((3, 128, 128))
    result_list = split_channels(data, 0)

    assert len(result_list) == 3
    for chan, layer in enumerate(result_list):
        assert layer[0].shape == (128, 128)
        assert (
            layer[1]['blending'] == 'translucent_no_depth'
            if chan == 0
            else 'additive'
        )


def test_split_channels_affine_nparray(kwargs):
    kwargs['affine'] = np.eye(3)
    data = np.zeros((3, 128, 128))
    result_list = split_channels(data, 0, **kwargs)

    assert len(result_list) == 3
    for d, meta, _ in result_list:
        assert d.shape == (128, 128)
        assert np.array_equal(meta['affine'], np.eye(3))


def test_split_channels_affine_napari(kwargs):
    kwargs['affine'] = Affine(affine_matrix=np.eye(3))
    data = np.zeros((3, 128, 128))
    result_list = split_channels(data, 0, **kwargs)

    assert len(result_list) == 3
    for d, meta, _ in result_list:
        assert d.shape == (128, 128)
        assert np.array_equal(meta['affine'].affine_matrix, np.eye(3))


def test_split_channels_multi_affine_napari(kwargs):
    kwargs['affine'] = [
        Affine(scale=[1, 1]),
        Affine(scale=[2, 2]),
        Affine(scale=[3, 3]),
    ]

    data = np.zeros((3, 128, 128))
    result_list = split_channels(data, 0, **kwargs)

    assert len(result_list) == 3
    for idx, result_data in enumerate(result_list):
        d, meta, _ = result_data
        assert d.shape == (128, 128)
        assert np.array_equal(
            meta['affine'].affine_matrix,
            Affine(scale=[idx + 1, idx + 1]).affine_matrix,
        )


@pytest.mark.parametrize(
    ('input_array', 'expected_type'),
    [
        (np.zeros((10, 20)), np.ndarray),
        (da.zeros((10, 20)), da.Array),
        (zarr.zeros((10, 20)), da.Array),
    ],
    ids=['numpy', 'dask', 'zarr'],
)
def test_images_to_stack_lazy_arrays(input_array, expected_type):
    """Test that images_to_stack handles numpy, dask, and zarr arrays correctly."""
    data = input_array
    images = [Image(data) for _ in range(3)]

    if isinstance(input_array, zarr.Array):
        with pytest.warns(
            UserWarning,
            match='zarr array cannot be stacked lazily, using dask array to stack.',
        ):
            stack = images_to_stack(images)
    else:
        stack = images_to_stack(images)

    assert not stack.multiscale
    assert isinstance(stack.data, expected_type)
    assert stack.data.shape[1:] == input_array.shape


@pytest.mark.parametrize(
    ('input_array', 'expected_type'),
    [
        (np.zeros((10, 20)), np.ndarray),
        (da.zeros((10, 20)), da.Array),
        (zarr.zeros((10, 20)), da.Array),
    ],
    ids=['numpy', 'dask', 'zarr'],
)
def test_images_to_stack_lazy_multiscale_arrays(input_array, expected_type):
    """Test stacking of multiscale numpy, dask, and zarr arrays."""
    # Slicing zarr array returns numpy array, so we need to re-wrap in zarr
    if isinstance(input_array, zarr.Array):
        data = [input_array, zarr.array(input_array[::2, ::2])]
    else:
        data = [input_array, input_array[::2, ::2]]

    images = [Image(data) for _ in range(3)]

    if isinstance(input_array, zarr.Array):
        with pytest.warns(
            UserWarning,
            match='zarr array cannot be stacked lazily, using dask array to stack.',
        ):
            stack = images_to_stack(images)
    else:
        stack = images_to_stack(images)

    assert stack.multiscale
    assert isinstance(stack.data[0], expected_type)
    assert stack.data[0].shape[1:] == input_array.shape


@pytest.mark.parametrize(
    ('array_type', 'expected_result_type'),
    [
        ('numpy', np.ndarray),
        ('dask', da.Array),
        ('zarr', da.Array),
    ],
)
def test_slice_from_axis_different_array_types(
    array_type, expected_result_type
):
    """Test slice_from_axis with numpy, dask, and zarr arrays."""
    # Create test data
    if array_type == 'numpy':
        data = np.zeros((3, 4, 4))
    elif array_type == 'dask':
        data = da.zeros((3, 4, 4))
    elif array_type == 'zarr':
        data = zarr.zeros((3, 4, 4))

    axis, element = 1, 2
    expected_shape = (3, 4)

    if array_type == 'zarr':
        with pytest.warns(
            UserWarning, match='zarr array cannot be sliced lazily'
        ):
            result = slice_from_axis(data, axis=axis, element=element)
    else:
        result = slice_from_axis(data, axis=axis, element=element)

    # Check result type and shape
    assert isinstance(result, expected_result_type)
    assert result.shape == expected_shape

    # Check result values - compare with expected slice
    result_computed = (
        result.compute() if hasattr(result, 'compute') else result
    )
    expected = data[:, element, :]
    np.testing.assert_array_equal(result_computed, expected)
