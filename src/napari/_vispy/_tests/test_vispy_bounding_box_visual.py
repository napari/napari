import sys
from itertools import product

import numpy as np
import pytest


@pytest.mark.skipif(
    sys.platform == 'win32', reason='This new test is flaky on windows'
)
def test_bounding_box_multiscale_3D(make_napari_viewer, qtbot):
    viewer = make_napari_viewer(show=True)

    data = np.ones((2, 200, 200))
    layer = viewer.add_image(
        [data, data[:, ::2, ::2], data[:, ::4, ::4]], multiscale=True
    )
    viewer.layers[-1].bounding_box.visible = True
    assert viewer.dims.ndisplay == 2

    # Set canvas size to target amount
    viewer.window._qt_viewer.canvas.size = (200, 200)
    viewer.window._qt_viewer.canvas.on_draw(None)
    viewer.camera.zoom = 2

    assert viewer.layers[0].data_level == 0

    viewer.dims.ndisplay = 3

    qtbot.waitUntil(lambda: viewer.layers[-1].loaded)
    # get the actual bounding box vertices
    bb = viewer.window._qt_viewer.canvas._layer_overlay_to_visual[layer][
        layer.bounding_box
    ]
    displayed_bbox_vertices = bb.node.markers._data['a_position'].astype(
        'float'
    )

    # The bounding box overlay uses level-0 extent transformed through
    # tile2data.inverse so it represents the full dataset correctly.
    displayed = viewer.dims.displayed
    bounds_lv0 = viewer.layers[-1]._display_bounding_box_at_level(
        displayed, 0
    ) + np.array([[-0.5, 0.5]])
    tile2data = viewer.layers[-1]._transforms[0]
    t2d_scale = np.asarray(tile2data.scale)[list(displayed)]
    t2d_translate = np.asarray(tile2data.translate)[list(displayed)]
    safe_scale = np.where(np.abs(t2d_scale) > 1e-12, t2d_scale, 1.0)
    expected_bounds = (bounds_lv0 - t2d_translate[:, np.newaxis]) / safe_scale[
        :, np.newaxis
    ]
    expected_vertices = np.array(list(product(*expected_bounds)))
    # roll the coordinates to match the vispy marker vertices
    expected_vertices = np.roll(expected_vertices, shift=-1, axis=1).astype(
        'float'
    )

    # the order of the vertices is not important, just the locations
    assert set(map(tuple, displayed_bbox_vertices)) == set(
        map(tuple, expected_vertices)
    )
