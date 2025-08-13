import numpy as np

from napari.components.overlays import (
    BoundingBoxOverlay,
    CanvasOverlay,
    ScaleBarOverlay,
)


def test_viewer_overlays(make_napari_viewer):
    viewer = make_napari_viewer()
    canvas = viewer.window._qt_viewer.canvas

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

    old_vispy_overlays = list(canvas._overlay_to_visual.values())

    new_overlay = ScaleBarOverlay()
    viewer._overlays['test'] = new_overlay

    assert new_overlay in canvas._overlay_to_visual
    new_overlay_node = canvas._overlay_to_visual[new_overlay][0].node
    assert new_overlay_node not in canvas.view.scene.children
    assert new_overlay_node in canvas.view.children

    # old visuals should be removed, as everything was recreated
    for old_ov in old_vispy_overlays:
        assert old_ov[0].node not in canvas.view.scene.children
        assert old_ov[0].node not in canvas.view.children

    viewer._overlays.pop('test')
    assert new_overlay not in canvas._overlay_to_visual
    assert new_overlay_node not in canvas.view.children


def test_layer_overlays(make_napari_viewer):
    viewer = make_napari_viewer()
    canvas = viewer.window._qt_viewer.canvas

    view_children = len(canvas.view.children)
    scene_children = len(canvas.view.scene.children)

    assert not canvas._layer_overlay_to_visual

    layer = viewer.add_points()
    layer_node = canvas.layer_to_visual[layer].node

    for overlay in layer._overlays.values():
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

    new_overlay = BoundingBoxOverlay()
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


def test_grid_mode(make_napari_viewer):
    viewer = make_napari_viewer(ndisplay=3)
    canvas = viewer.window._qt_viewer.canvas

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
