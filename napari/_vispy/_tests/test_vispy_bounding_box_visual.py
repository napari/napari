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
    viewer.add_image(
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

    qtbot.waitUntil(lambda: viewer.layers[-1]._loaded)
    # get the actual bounding box vertices
    displayed_bbox_vertices = (
        viewer.window._qt_viewer.canvas.view.scene.children[4]
        .children[2]
        .markers._data['a_position']
        .astype('float')
    )

    # for multiscale layers, in 3D, the lowest data level is displayed
    # get the vertices from the lowest data level bounding box, augmented
    expected_vertices = np.array(
        list(
            product(
                *viewer.layers[-1]._display_bounding_box_at_level(
                    viewer.dims.displayed, len(viewer.layers[-1].data) - 1
                )
                + np.array([[-0.5, 0.5]])
            )
        )
    )
    # roll the coordinates to match the vispy marker vertices
    expected_vertices = np.roll(expected_vertices, shift=-1, axis=1).astype(
        'float'
    )

    # the order of the vertices is not important, just the locations
    assert set(map(tuple, displayed_bbox_vertices)) == set(
        map(tuple, expected_vertices)
    )
