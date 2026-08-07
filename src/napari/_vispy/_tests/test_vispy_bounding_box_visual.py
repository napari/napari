import sys
from itertools import product

import numpy as np
import pytest


def _expected_multiscale_vertices(layer, displayed):
    """Vertices the bounding box node should have in its own (tile) frame.

    The overlay converts the level-0 pixel-edge extent ``[0, shape]`` into
    tile coordinates by dividing by the displayed level's downsampling
    scale, then shifting by half a pixel to pixel-center coordinates. The
    corner-pixel translation is intentionally *not* inverted: the child
    offset applied in ``VispyBaseLayer._on_matrix_change`` cancels it.
    """
    bounds = layer._display_bounding_box_at_level(
        list(displayed), 0
    ) + np.array([[0.0, 1.0]])
    scale = np.asarray(layer._transforms[0].scale)[list(displayed)]
    bounds = bounds / scale[:, np.newaxis] - 0.5
    vertices = np.array(list(product(*bounds)))
    # roll the coordinates to match the vispy marker vertices
    return np.roll(vertices, shift=-1, axis=1).astype('float')


def _bounding_box_world_vertices(viewer, layer):
    """Map the bounding box marker vertices into scene (world) coordinates."""
    bb = viewer.window._qt_viewer.canvas._layer_overlay_to_visual[layer][
        layer.bounding_box
    ]
    vertices = bb.node.markers._data['a_position'].astype(float)
    transform = bb.node.node_transform(
        viewer.window._qt_viewer.canvas.view.scene
    )
    return transform.map(vertices)


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

    expected_vertices = _expected_multiscale_vertices(
        viewer.layers[-1], viewer.dims.displayed
    )

    # the order of the vertices is not important, just the locations
    assert set(map(tuple, displayed_bbox_vertices)) == set(
        map(tuple, expected_vertices)
    )


@pytest.mark.skipif(
    sys.platform == 'win32', reason='This new test is flaky on windows'
)
def test_bounding_box_multiscale_2D_zoom_stable(make_napari_viewer, qtbot):
    """The box must stay at the full dataset extent while zooming.

    Zooming a 2D multiscale layer changes the displayed data level and the
    corner pixels; the bounding box previously drifted/contracted because
    the corner offset was compensated twice (napari/napari#9142).
    """
    viewer = make_napari_viewer(show=True)

    data = np.ones((800, 800))
    layer = viewer.add_image(
        [data, data[::2, ::2], data[::4, ::4]], multiscale=True
    )
    layer.bounding_box.visible = True

    viewer.window._qt_viewer.canvas.size = (200, 200)

    # the full level-0 extent, from pixel edge to pixel edge
    expected_range = (-0.5, 799.5)

    seen_states = set()
    for zoom in (0.2, 1.0, 4.0):
        viewer.camera.zoom = zoom
        viewer.camera.center = (0.0, 300.0, 300.0)
        viewer.window._qt_viewer.canvas.on_draw(None)
        qtbot.waitUntil(lambda: layer.loaded)
        viewer.window._qt_viewer.canvas.on_draw(None)

        seen_states.add((layer.data_level, tuple(layer.corner_pixels.ravel())))

        world = _bounding_box_world_vertices(viewer, layer)
        for axis in range(2):
            np.testing.assert_allclose(
                (world[:, axis].min(), world[:, axis].max()),
                expected_range,
                err_msg=f'zoom={zoom}, axis={axis}',
            )

    # make sure zooming actually changed the level/corners under test
    assert len(seen_states) > 1
