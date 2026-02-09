import numpy as np

from napari.components.overlays import (
    AxesOverlay,
    BoundingBoxOverlay,
    CanvasOverlay,
    ScaleBarOverlay,
)


def test_scene_overlays(qt_viewer):
    viewer = qt_viewer.viewer
    vispy_canvas = qt_viewer.canvas

    for overlay in viewer._overlays.values():
        # vispy overlays only exist if they are visible at least once
        overlay.visible = True
        assert (
            vispy_canvas._scene_overlay_to_visual[overlay].node
            in vispy_canvas.view.scene.children
        )

    old_vispy_scene_overlays = dict(
        vispy_canvas._scene_overlay_to_visual.items()
    )

    new_overlay = AxesOverlay(visible=True)
    viewer._overlays['test'] = new_overlay

    assert new_overlay in vispy_canvas._scene_overlay_to_visual
    new_overlay_node = vispy_canvas._scene_overlay_to_visual[new_overlay].node
    assert new_overlay_node in vispy_canvas.view.scene.children
    assert new_overlay_node not in vispy_canvas.view.children

    # old visuals should still be there, as they are reused when possible
    for _, vispy_overlay in old_vispy_scene_overlays.items():
        assert vispy_overlay.node in vispy_canvas.view.scene.children

    viewer._overlays.pop('test')
    assert new_overlay not in vispy_canvas._scene_overlay_to_visual
    assert new_overlay_node not in vispy_canvas.view.children


def test_canvas_overlays(qt_viewer):
    canvas = qt_viewer.viewer.canvas
    vispy_canvas = qt_viewer.canvas

    for overlay in canvas._overlays.values():
        # vispy overlays only exist if they are visible at least once
        overlay.visible = True
        assert all(
            visual.node in vispy_canvas.view.children
            for visual in vispy_canvas._canvas_overlay_to_visual[overlay]
        )

    old_vispy_canvas_overlays = {
        k: list(v) for k, v in vispy_canvas._canvas_overlay_to_visual.items()
    }

    new_overlay = ScaleBarOverlay(visible=True)
    canvas._overlays['test'] = new_overlay

    assert new_overlay in vispy_canvas._canvas_overlay_to_visual
    new_overlay_node = vispy_canvas._canvas_overlay_to_visual[new_overlay][
        0
    ].node
    assert new_overlay_node not in vispy_canvas.view.scene.children
    assert new_overlay_node in vispy_canvas.view.children

    # old visuals should still be there, as they are reused when possible
    for _, vispy_overlays in old_vispy_canvas_overlays.items():
        for vispy_overlay in vispy_overlays:
            assert vispy_overlay.node in vispy_canvas.view.children

    canvas._overlays.pop('test')
    assert new_overlay not in vispy_canvas._canvas_overlay_to_visual
    assert new_overlay_node not in vispy_canvas.view.children

    canvas.welcome.visible = False  # just for proper test cleanup


def test_layer_overlays(qt_viewer):
    viewer = qt_viewer.viewer
    canvas = qt_viewer.canvas

    view_children = len(canvas.view.children)
    scene_children = len(canvas.view.scene.children)

    assert not canvas._layer_overlay_to_visual

    layer = viewer.add_points()
    layer_node = canvas.layer_to_visual[layer].node

    for overlay in layer._overlays.values():
        # vispy overlays only exist if they are visible at least once
        overlay.visible = True
        if isinstance(overlay, CanvasOverlay):
            assert (
                canvas._layer_overlay_to_visual[layer][overlay].node
                in canvas.view.children
            )
        else:
            assert (
                canvas._layer_overlay_to_visual[layer][overlay].node
                in layer_node.children
            )

    old_vispy_overlays = {**canvas._layer_overlay_to_visual[layer]}

    new_overlay = BoundingBoxOverlay(visible=True)
    layer._overlays['test'] = new_overlay

    assert new_overlay in canvas._layer_overlay_to_visual[layer]
    new_overlay_node = canvas._layer_overlay_to_visual[layer][new_overlay].node
    assert new_overlay_node in layer_node.children
    assert new_overlay_node not in canvas.view.children

    # old visuals should still be there, as they are reused when possible
    for overlay, vispy_overlay in old_vispy_overlays.items():
        if isinstance(overlay, CanvasOverlay):
            assert vispy_overlay.node in canvas.view.children
        else:
            assert vispy_overlay.node in layer_node.children

    layer._overlays.pop('test')
    assert new_overlay not in canvas._layer_overlay_to_visual[layer]
    assert new_overlay_node not in canvas.view.children

    viewer.layers.pop()

    # should be back to the status quo
    assert not canvas._layer_overlay_to_visual
    assert len(canvas.view.children) == view_children
    assert len(canvas.view.scene.children) == scene_children


def test_grid_mode(qt_viewer):
    viewer = qt_viewer.viewer
    canvas = qt_viewer.canvas

    viewer.dims.ndisplay = 3
    viewer.add_image(np.ones((10, 10, 10)))

    angles = 10, 20, 30  # just some nonzero stuff
    zoom = 1
    viewer.camera.angles = angles
    viewer.camera.zoom = zoom

    canvas.on_draw(None)

    for camera in (canvas.camera, *canvas.grid_cameras):
        np.testing.assert_allclose(camera.angles, angles)
        assert camera.zoom == zoom

    # ensure that switching to grid maintains zoom and angles
    viewer.canvas.grid.enabled = True

    canvas.on_draw(None)

    for camera in (canvas.camera, *canvas.grid_cameras):
        np.testing.assert_allclose(camera.angles, angles)
        assert camera.zoom == zoom

    viewer.canvas.grid.enabled = False

    canvas.on_draw(None)

    for camera in (canvas.camera, *canvas.grid_cameras):
        np.testing.assert_allclose(camera.angles, angles)
        assert camera.zoom == zoom


def test_tiling_canvas_overlays(qt_viewer):
    viewer = qt_viewer.viewer
    canvas = qt_viewer.canvas

    viewer.canvas.scale_bar.visible = True
    viewer.canvas.text.visible = True
    viewer.canvas.text.text = 'test'
    viewer.canvas.scale_bar.position = 'bottom_left'
    viewer.canvas.text_overlay.position = 'bottom_left'

    vispy_scale_bar = canvas._canvas_overlay_to_visual[
        viewer.canvas.scale_bar
    ][0]
    vispy_text_overlay = canvas._canvas_overlay_to_visual[viewer.canvas.text][
        0
    ]

    padding = 10.0  # currently hardcoded
    y_max, x_max = canvas.size

    scale_bar_y_size = vispy_scale_bar.y_size + padding
    scale_bar_x_size = vispy_scale_bar.x_size + padding

    text_overlay_y_size = vispy_text_overlay.y_size + padding
    text_overlay_x_size = vispy_text_overlay.x_size + padding

    # check vertical tiling works on the bottom right
    viewer.canvas.scale_bar.position = 'bottom_right'
    viewer.canvas.text.position = 'bottom_right'
    canvas._update_overlay_canvas_positions()

    np.testing.assert_almost_equal(
        vispy_text_overlay.node.transform.translate[0],
        x_max - text_overlay_x_size,
        decimal=3,
    )
    np.testing.assert_almost_equal(
        vispy_text_overlay.node.transform.translate[1],
        y_max - text_overlay_y_size - scale_bar_y_size,
        decimal=3,
    )

    # move scale bar out of the way and check tiling is updated
    viewer.canvas.scale_bar.position = 'top_right'
    canvas._update_overlay_canvas_positions()
    np.testing.assert_almost_equal(
        vispy_text_overlay.node.transform.translate[0],
        x_max - text_overlay_x_size,
        decimal=3,
    )
    np.testing.assert_almost_equal(
        vispy_text_overlay.node.transform.translate[1],
        y_max - text_overlay_y_size,
        decimal=3,
    )

    # check horizontal tiling works on the top right
    viewer.canvas.text.position = 'top_right'
    canvas._update_overlay_canvas_positions()
    np.testing.assert_almost_equal(
        vispy_text_overlay.node.transform.translate[0],
        x_max - text_overlay_x_size - scale_bar_x_size,
        decimal=3,
    )
    np.testing.assert_almost_equal(
        vispy_text_overlay.node.transform.translate[1],
        0 + padding,
        decimal=3,
    )
