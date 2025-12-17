import numpy as np

from napari.components.overlays import (
    BoundingBoxOverlay,
    CanvasOverlay,
    ScaleBarOverlay,
)


def test_viewer_overlays(qt_viewer):
    viewer = qt_viewer.viewer
    canvas = qt_viewer.canvas

    for overlay in viewer._overlays.values():
        if isinstance(overlay, CanvasOverlay):
            assert all(
                visual.node in canvas.view.children
                for visual in canvas._overlay_to_visual[overlay]
            )
        else:
            assert all(
                visual.node in canvas.view.scene.children
                for visual in canvas._overlay_to_visual[overlay]
            )

    new_overlay = ScaleBarOverlay()
    viewer._overlays['test'] = new_overlay

    assert new_overlay in canvas._overlay_to_visual
    new_overlay_node = canvas._overlay_to_visual[new_overlay][0].node
    assert new_overlay_node not in canvas.view.scene.children
    assert new_overlay_node in canvas.view.children

    viewer._overlays.pop('test')
    assert new_overlay not in canvas._overlay_to_visual
    assert new_overlay_node not in canvas.view.children


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

    # old visuals should be removed, as everything was recreated
    for old_ov in old_vispy_overlays.values():
        assert old_ov.node not in canvas.view.scene.children
        assert old_ov.node not in canvas.view.children

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
    viewer.grid.enabled = True

    canvas.on_draw(None)

    for camera in (canvas.camera, *canvas.grid_cameras):
        np.testing.assert_allclose(camera.angles, angles)
        assert camera.zoom == zoom

    viewer.grid.enabled = False

    canvas.on_draw(None)

    for camera in (canvas.camera, *canvas.grid_cameras):
        np.testing.assert_allclose(camera.angles, angles)
        assert camera.zoom == zoom


def test_tiling_canvas_overlays(qt_viewer):
    viewer = qt_viewer.viewer
    canvas = qt_viewer.canvas

    viewer.scale_bar.visible = True
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = 'test'

    vispy_scale_bar = canvas._overlay_to_visual[viewer.scale_bar][0]
    vispy_text_overlay = canvas._overlay_to_visual[viewer.text_overlay][0]

    padding = 10.0  # currently hardcoded
    y_max, x_max = canvas.size

    scale_bar_y_size = vispy_scale_bar.y_size + padding
    scale_bar_x_size = vispy_scale_bar.x_size + padding

    text_overlay_y_size = vispy_text_overlay.y_size + padding
    text_overlay_x_size = vispy_text_overlay.x_size + padding

    # check vertical tiling works on the bottom right
    viewer.scale_bar.position = 'bottom_right'
    viewer.text_overlay.position = 'bottom_right'
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
    viewer.scale_bar.position = 'top_right'
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
    viewer.text_overlay.position = 'top_right'
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
