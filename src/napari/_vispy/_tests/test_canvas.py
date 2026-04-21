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
        # vispy overlays only exist if they are visible at least once
        overlay.visible = True
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

    old_vispy_overlays = {
        k: list(v) for k, v in canvas._overlay_to_visual.items()
    }

    new_overlay = ScaleBarOverlay(visible=True)
    viewer._overlays['test'] = new_overlay

    assert new_overlay in canvas._overlay_to_visual
    new_overlay_node = canvas._overlay_to_visual[new_overlay][0].node
    assert new_overlay_node not in canvas.view.scene.children
    assert new_overlay_node in canvas.view.children

    # old visuals should still be there, as they are reused when possible
    for overlay, vispy_overlays in old_vispy_overlays.items():
        for vispy_overlay in vispy_overlays:
            if isinstance(overlay, CanvasOverlay):
                assert vispy_overlay.node in canvas.view.children
            else:
                assert vispy_overlay.node in canvas.view.scene.children

    viewer._overlays.pop('test')
    assert new_overlay not in canvas._overlay_to_visual
    assert new_overlay_node not in canvas.view.children

    viewer.welcome_screen.visible = False  # just for proper test cleanup


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
    viewer.scale_bar.position = 'bottom_left'
    viewer.text_overlay.position = 'bottom_left'

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


def test_first_viewer_overlay_visible_event_reaches_listener(qt_viewer, qtbot):
    viewer = qt_viewer.viewer
    canvas = qt_viewer.canvas

    calls = []
    viewer.scale_bar.events.visible.connect(lambda: calls.append('visible'))

    assert not viewer.scale_bar.visible
    assert (
        viewer.scale_bar.events.visible._slot_index(
            canvas._update_viewer_overlays
        )
        != -1
    )

    viewer.scale_bar.visible = True

    assert calls == ['visible']

    # Let the event loop process the deferred disconnect scheduled with
    # QTimer.singleShot(0, ...).
    qtbot.wait(0)

    assert (
        viewer.scale_bar.events.visible._slot_index(
            canvas._update_viewer_overlays
        )
        == -1
    )


def test_first_layer_overlay_visible_event_reaches_listener(qt_viewer, qtbot):
    viewer = qt_viewer.viewer
    canvas = qt_viewer.canvas
    layer = viewer.add_points()
    overlay = next(iter(layer._overlays.values()))

    calls = []
    overlay.events.visible.connect(lambda: calls.append('visible'))

    assert not overlay.visible
    assert (
        overlay.events.visible._slot_index(canvas._overlay_callbacks[layer])
        != -1
    )

    overlay.visible = True

    assert calls == ['visible']

    # Let the event loop process the deferred disconnect scheduled with
    # QTimer.singleShot(0, ...).
    qtbot.wait(0)

    assert (
        overlay.events.visible._slot_index(canvas._overlay_callbacks[layer])
        == -1
    )


def test_world_units_restored_after_removing_inconsistent_layer(qt_viewer):
    """Removing a units-inconsistent layer should re-enable unit-aware rendering.

    Regression test for #8771: when a layer with pixel (dimensionless) units is added to
    a viewer that already has length-unit layers, world units become inconsistent
    and unit-aware rendering is disabled.Upon removing the inconsistent layer,
    the canvas must call _update_world_units() on the next draw so the remaining compatible
    layers resume using the shared world units.
    """
    from pint import get_application_registry

    reg = get_application_registry()

    viewer = qt_viewer.viewer
    canvas = qt_viewer.canvas

    # Two images with compatible length units (um and nm share the same
    # dimensionality; consistent world units = nm, the smaller one).
    im1 = viewer.add_image(np.zeros((10, 10)), units=('um', 'um'))
    viewer.add_image(np.zeros((10, 10)), units=('nm', 'nm'))

    # Units should be consistent after adding compatible layers.
    assert viewer.layers.extent.units is not None
    vispy_im1 = canvas.layer_to_visual[im1]

    # the vispy layer received the shared world units (nm). (not just im1's own layer units (um))
    assert vispy_im1._world_units == (reg.nm, reg.nm)

    # Add layer with incompatible units (pixels)
    labels = viewer.add_labels(np.zeros((10, 10), dtype=int))

    # Units are now inconsistent across layers; _update_world_units() sets
    # world_units=None on each vispy layer, which causes the vispy layer's
    # _world_units to fall back to its own layer-local units.
    assert viewer.layers.extent.units is None
    assert vispy_im1._world_units == (
        reg.um,
        reg.um,
    )  # im1's own layer units, not (nm, nm)

    # Remove the incompatible layer.
    viewer.layers.remove(labels)

    canvas.on_draw(None)
    assert vispy_im1.world_units == (reg.nm, reg.nm)


def test_world_units_applied_to_inserted_layer_via_layerlist_event(qt_viewer):
    from pint import get_application_registry

    reg = get_application_registry()

    viewer = qt_viewer.viewer
    canvas = qt_viewer.canvas

    viewer.add_image(np.zeros((10, 10)), units=('um', 'um'))
    image_nm = viewer.add_image(np.zeros((10, 10)), units=('nm', 'nm'))

    assert viewer.layers.extent.units == (reg.nm, reg.nm)
    assert canvas.layer_to_visual[image_nm].world_units == (reg.nm, reg.nm)


def test_inserted_layer_receives_shared_world_units_when_units_unchanged(
    qt_viewer,
):
    from pint import get_application_registry

    reg = get_application_registry()

    viewer = qt_viewer.viewer
    canvas = qt_viewer.canvas

    viewer.add_image(np.zeros((10, 10)), units=('nm', 'nm'))
    image_um = viewer.add_image(np.zeros((10, 10)), units=('um', 'um'))

    assert viewer.layers.extent.units == (reg.nm, reg.nm)
    assert canvas.layer_to_visual[image_um].world_units == (reg.nm, reg.nm)
