import itertools

import numpy as np
import pytest
from vispy.color import Colormap
from vispy.visuals.filters import Alpha

from napari._vispy.layers.tiled_image import TiledImageNode


def test_big_2D_image(make_napari_viewer):
    """Test big 2D image with axis exceeding max texture size."""
    viewer = make_napari_viewer()

    shape = (20_000, 10)
    data = np.random.random(shape)
    layer = viewer.add_image(data, multiscale=False)
    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]
    assert visual.node is not None
    if (
        visual.MAX_TEXTURE_SIZE_2D is not None
        and shape[0] > visual.MAX_TEXTURE_SIZE_2D
    ):
        assert isinstance(visual.node, TiledImageNode)
        assert visual.MAX_TEXTURE_SIZE_2D in itertools.chain(
            *visual.node.offsets
        )


def test_big_3D_image(make_napari_viewer):
    """Test big 3D image with axis exceeding max texture size."""
    viewer = make_napari_viewer(ndisplay=3)

    shape = (5, 10, 3_000)
    data = np.random.random(shape)
    layer = viewer.add_image(data, multiscale=False)
    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]
    assert visual.node is not None
    if visual.MAX_TEXTURE_SIZE_3D is not None:
        s = np.ceil(np.divide(shape, visual.MAX_TEXTURE_SIZE_3D)).astype(int)
        np.testing.assert_array_equal(layer._transforms['tile2data'].scale, s)


@pytest.mark.parametrize(
    'shape',
    [(2, 4), (256, 4048), (4, 20_000), (20_000, 4)],
)
def test_downsample_value(make_napari_viewer, shape):
    """Test getting correct value for downsampled data."""
    viewer = make_napari_viewer()

    data = np.zeros(shape)
    data[shape[0] // 2 :, shape[1] // 2 :] = 1
    layer = viewer.add_image(data, multiscale=False)

    test_points = [
        (int(shape[0] * 0.25), int(shape[1] * 0.25)),
        (int(shape[0] * 0.75), int(shape[1] * 0.25)),
        (int(shape[0] * 0.25), int(shape[1] * 0.75)),
        (int(shape[0] * 0.75), int(shape[1] * 0.75)),
    ]
    expected_values = [0.0, 0.0, 0.0, 1.0]

    for test_point, expected_value in zip(
        test_points, expected_values, strict=False
    ):
        viewer.cursor.position = test_point
        assert (
            layer.get_value(viewer.cursor.position, world=True)
            == expected_value
        )


def test_tiled_image_keeps_settings_when_tile_count_changes():
    node = TiledImageNode(np.zeros((4, 4), dtype=np.float32), tile_size=2)
    cmap = Colormap([[0, 0, 0, 1], [1, 0, 0, 1]])
    node.cmap = cmap
    node.clim = (0.2, 0.8)
    node.set_gl_state(blend=True, blend_func=('one', 'one'))
    filt = Alpha(0.5)
    node.attach(filt)

    node.set_data(np.zeros((0, 0), dtype=np.float32))
    node.set_data(np.zeros((6, 6), dtype=np.float32))

    assert len(node.adopted_children) == 9
    for child in node.adopted_children:
        assert child.cmap is cmap
        assert child.clim == (0.2, 0.8)
        assert child._vshare.gl_state['blend_func'] == ('one', 'one')
        assert filt in child._filters
    node.detach(filt)
