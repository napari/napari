import numpy as np
import pytest
from vispy.scene import PanZoomCamera, SceneCanvas

from napari._vispy.layers.tiled_image import TiledImageNode
from napari._vispy.visuals.image import Image as ImageNode
from napari.utils.colormaps import Colormap
from napari.utils.colormaps.colormap_utils import _napari_cmap_to_vispy

RED = [1, 0, 0, 1]
BLACK_WHITE = [[0, 0, 0, 1], [1, 1, 1, 1]]
BACKGROUND = 'magenta'  # no colormap under test produces it


def render(node_factory, nan_color=RED):
    """Render `[0.5, NaN, 0.5]` and return the RGB of each value."""
    values = [0.5, np.nan, 0.5]
    data = np.tile(np.array(values, dtype=np.float32), (4, 1))
    cmap = _napari_cmap_to_vispy(
        Colormap(BLACK_WHITE, name='testing', nan_color=nan_color),
        decode_nan_sentinel=True,
    )
    canvas = SceneCanvas(size=(48, 48), show=False, bgcolor=BACKGROUND)
    try:
        view = canvas.central_widget.add_view()
        node_factory(data, view, cmap)
        view.camera = PanZoomCamera(aspect=1)
        view.camera.set_range(x=(0, len(values)), y=(0, 4), margin=0)
        img = canvas.render(alpha=True)
    finally:
        canvas.close()

    drawn = ~np.all(
        img == np.array([255, 0, 255, 255], dtype=np.uint8), axis=-1
    )
    ys, xs = np.where(drawn)
    assert xs.size, 'nothing was drawn'
    x0, x1 = xs.min(), xs.max() + 1
    row = (ys.min() + ys.max()) // 2
    return [
        tuple(
            int(c) for c in img[row, int(x0 + (i + 0.5) * (x1 - x0) / 3)][:3]
        )
        for i in range(3)
    ]


def _plain(data, view, cmap):
    ImageNode(
        data,
        cmap=cmap,
        clim=(0.0, 1.0),
        interpolation='nearest',
        texture_format='auto',
        parent=view.scene,
    )


def _tiled(data, view, cmap):
    node = TiledImageNode(data, tile_size=2, texture_format='auto')
    node.parent = view.scene
    node.cmap = cmap
    node.clim = (0.0, 1.0)
    node.interpolation = 'nearest'


@pytest.mark.usefixtures('qapp')
@pytest.mark.parametrize('node', [_plain, _tiled], ids=['image', 'tiled'])
def test_nan_renders_as_nan_color(node):
    _, nan_pixel, _ = render(node)
    assert nan_pixel == (255, 0, 0)
