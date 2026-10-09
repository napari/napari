"""Tests for ViewerModel.canvas_to_world and ViewerModel.world_to_canvas."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from napari.components import ViewerModel
from napari.utils.transforms import Affine


def _make_2d_viewer(canvas_size=(800, 600), center=(0, 7, 9.5), zoom=1.0):
    viewer = ViewerModel(ndisplay=2)
    viewer.add_image(np.ones((10, 15, 20)))
    viewer.canvas.size = canvas_size
    viewer.scene.camera.center = center
    viewer.scene.camera.zoom = zoom
    return viewer


def _make_3d_viewer(
    canvas_size=(800, 600),
    center=(5.0, 5.0, 5.0),
    zoom=1.0,
    angles=(0, 0, 0),
    perspective=0,
):
    viewer = ViewerModel(ndisplay=3)
    viewer.add_image(np.ones((10, 10, 10)))
    viewer.canvas.size = canvas_size
    viewer.scene.camera.center = center
    viewer.scene.camera.zoom = zoom
    viewer.scene.camera.angles = angles
    viewer.scene.camera.perspective = perspective
    return viewer


def test_2d_center_maps_to_viewbox_center():
    viewer = _make_2d_viewer()
    viewbox_size = np.array(viewer.canvas.viewbox_size(viewer.layers))
    # camera center (displayed part) should project to viewbox center
    world = np.array(viewer.scene.camera.center)
    canvas = viewer.world_to_canvas(world)
    np.testing.assert_allclose(canvas, viewbox_size / 2)


def test_2d_known_values():
    viewer = _make_2d_viewer(center=(0, 5, 5), zoom=2.0)
    viewbox_size = np.array(viewer.canvas.viewbox_size(viewer.layers))
    vc = viewbox_size / 2
    # world = center + (canvas - vc) / zoom
    canvas_pos = (100, 200)
    world = viewer.canvas_to_world(canvas_pos)
    expected_displayed = (np.array(canvas_pos) - vc) / 2.0 + np.array([5, 5])
    np.testing.assert_allclose(
        world[list(viewer.dims.displayed)], expected_displayed
    )
    # and back
    np.testing.assert_allclose(viewer.world_to_canvas(world), canvas_pos)


def test_2d_zoom_translation():
    viewer = _make_2d_viewer(center=(0, 0, 0), zoom=1.0)
    viewer.scene.camera.zoom = 3.0
    viewer.scene.camera.center = (0, 10, 20)
    # one world unit = zoom canvas pixels
    c0 = viewer.world_to_canvas(np.array([0, 10, 20]))
    c1 = viewer.world_to_canvas(np.array([0, 11, 20]))
    np.testing.assert_allclose(c1 - c0, [3.0, 0.0])
    c2 = viewer.world_to_canvas(np.array([0, 10, 22]))
    np.testing.assert_allclose(c2 - c0, [0.0, 6.0])


def test_2d_roundtrip_single_and_batch():
    viewer = _make_2d_viewer(center=(0, 4, 5), zoom=1.7)
    rng = np.random.default_rng(0)
    canvases = rng.uniform(low=[0, 0], high=[800, 600], size=(10, 2))
    for cw in canvases:
        world = viewer.canvas_to_world(tuple(cw))
        back = viewer.world_to_canvas(world)
        np.testing.assert_allclose(back, cw, rtol=1e-10, atol=1e-8)
    # batched world -> canvas
    worlds = np.array([viewer.canvas_to_world(tuple(cw)) for cw in canvases])
    backs = viewer.world_to_canvas(worlds)
    assert backs.shape == (10, 2)
    np.testing.assert_allclose(backs, canvases, rtol=1e-10, atol=1e-8)
    # single returns shape (2,)
    assert viewer.world_to_canvas(worlds[0]).shape == (2,)


def test_2d_nondisplayed_ignored_and_preserved():
    viewer = _make_2d_viewer()
    viewer.dims.set_point(0, 3.0)
    world = np.array([123.0, 5.0, 5.0])
    canvas = viewer.world_to_canvas(world)
    # non-displayed dim should not affect projection
    world2 = np.array([999.0, 5.0, 5.0])
    np.testing.assert_allclose(viewer.world_to_canvas(world2), canvas)
    # canvas_to_world fills non-displayed dims from dims.point
    back = viewer.canvas_to_world(tuple(canvas))
    assert back[0] == pytest.approx(3.0)
    np.testing.assert_allclose(back[1:], [5.0, 5.0])


def test_2d_viewbox_offset():
    viewer = _make_2d_viewer()
    viewer.add_image(np.ones((5, 5)))
    viewer.canvas.grid.enabled = True
    viewbox_size = np.array(viewer.canvas.viewbox_size(viewer.layers))
    world = np.array(viewer.scene.camera.center)
    c00 = viewer.world_to_canvas(world, viewbox=(0, 0))
    c01 = viewer.world_to_canvas(world, viewbox=(0, 1))
    c10 = viewer.world_to_canvas(world, viewbox=(1, 0))
    np.testing.assert_allclose(c00, viewbox_size / 2)
    # (row, col): col shifts x, row shifts y
    np.testing.assert_allclose(c01, viewbox_size / 2 + [0, viewbox_size[1]])
    np.testing.assert_allclose(c10, viewbox_size / 2 + [viewbox_size[0], 0])
    # roundtrip per viewbox
    for vb in [(0, 0), (0, 1), (1, 0)]:
        cw = (10.0, 20.0)
        w = viewer.canvas_to_world(cw, viewbox=vb)
        np.testing.assert_allclose(viewer.world_to_canvas(w, viewbox=vb), cw)


@pytest.mark.parametrize(
    'angles',
    [(0, 0, 0), (30, 45, 60), (90, 0, 0), (10, 20, 30), (-45, 15, 70)],
)
@pytest.mark.parametrize('zoom', [0.5, 1.0, 2.5])
def test_3d_ortho_roundtrip_canvas_world(angles, zoom):
    viewer = _make_3d_viewer(center=(5, -3, 12), zoom=zoom, angles=angles)
    rng = np.random.default_rng(hash(angles) % 2**32)
    for _ in range(5):
        cw = (float(rng.uniform(0, 800)), float(rng.uniform(0, 600)))
        world = viewer.canvas_to_world(cw)
        back = viewer.world_to_canvas(world)
        np.testing.assert_allclose(back, cw, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize('angles', [(0, 0, 0), (30, 45, 60), (10, 20, 30)])
def test_3d_ortho_roundtrip_world_canvas(angles):
    # canvas_to_world projects onto the camera-center plane (depth 0),
    # so arbitrary off-plane worlds lose their depth component.
    # Check instead that:
    # 1. on-plane worlds (from canvas_to_world) roundtrip exactly, and
    # 2. arbitrary worlds are stable: w -> c -> w' -> c' with c == c'.
    viewer = _make_3d_viewer(center=(1, 2, 3), zoom=1.5, angles=angles)
    rng = np.random.default_rng(42)
    # 1. on-plane roundtrip
    for _ in range(5):
        cw = (float(rng.uniform(0, 800)), float(rng.uniform(0, 600)))
        on_plane = viewer.canvas_to_world(cw)
        back = viewer.canvas_to_world(tuple(viewer.world_to_canvas(on_plane)))
        np.testing.assert_allclose(back, on_plane, rtol=1e-10, atol=1e-8)
    # 2. stability for arbitrary worlds
    worlds = rng.normal(loc=5, scale=20, size=(10, 3))
    canvases = viewer.world_to_canvas(worlds)
    assert canvases.shape == (10, 2)
    for _c in canvases:
        w_prime = viewer.canvas_to_world(tuple(_c))
        c_prime = viewer.world_to_canvas(w_prime)
        np.testing.assert_allclose(c_prime, _c, rtol=1e-10, atol=1e-8)


def test_3d_camera_center_projects_to_viewbox_center():
    viewer = _make_3d_viewer(center=(5, 5, 5), angles=(30, 20, 10))
    viewbox_size = np.array(viewer.canvas.viewbox_size(viewer.layers))
    canvas = viewer.world_to_canvas(np.array([5, 5, 5]))
    np.testing.assert_allclose(canvas, viewbox_size / 2)


def test_3d_ortho_matches_affine():
    """The orthographic part is exactly an Affine; check equivalence.

    This is the basis for replacing the manual math with Transform objects.
    """
    viewer = _make_3d_viewer(center=(1, 2, 3), zoom=2.0, angles=(25, -30, 45))
    viewbox_size = np.array(viewer.canvas.viewbox_size(viewer.layers))
    vc3 = np.array([0, *viewbox_size / 2])
    center = np.array(viewer.scene.camera.center)
    rot = R.from_euler(
        'xyz', np.asarray(viewer.scene.camera.angles), degrees=True
    ).as_matrix()
    linear = rot.T / viewer.scene.camera.zoom
    translate = center - rot.T @ vc3 / viewer.scene.camera.zoom
    mat = np.eye(4)
    mat[:-1, :-1] = linear
    mat[:-1, -1] = translate
    aff = Affine(affine_matrix=mat)
    rng = np.random.default_rng(1)
    for _ in range(5):
        cw = (float(rng.uniform(0, 800)), float(rng.uniform(0, 600)))
        expected = viewer.canvas_to_world(cw)
        got = aff(np.array([0, *cw]))
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-10)
        # inverse affine == world_to_canvas (displayed part)
        back_affine_3d = aff.inverse(expected)
        back = viewer.world_to_canvas(expected)
        np.testing.assert_allclose(
            back_affine_3d[1:], back, rtol=1e-10, atol=1e-8
        )


def test_2d_matches_scale_translate():
    from napari.utils.transforms import ScaleTranslate

    viewer = _make_2d_viewer(center=(0, 5, 7), zoom=2.0)
    viewbox_size = np.array(viewer.canvas.viewbox_size(viewer.layers))
    vc = viewbox_size / 2
    cc = np.array(viewer.scene.camera.center)[-2:]
    aff = ScaleTranslate(scale=[1 / 2.0] * 2, translate=cc - vc / 2.0)
    canvas = (123.0, 321.0)
    np.testing.assert_allclose(
        aff(np.array(canvas)),
        viewer.canvas_to_world(canvas)[list(viewer.dims.displayed)],
    )
    np.testing.assert_allclose(aff.inverse(np.array(cc)), vc)


def test_viewbox_to_world_is_persistent_affine():
    from napari.utils.transforms import Affine

    viewer2d = _make_2d_viewer()
    tr2d = viewer2d.viewbox_to_world
    assert isinstance(tr2d, Affine)
    assert tr2d.ndim == 2
    assert tr2d.linear_matrix.shape == (2, 2)
    assert tr2d.translate.shape == (2,)
    assert tr2d.name == 'viewbox_to_world'

    viewer3d = _make_3d_viewer()
    tr3d = viewer3d.viewbox_to_world
    assert isinstance(tr3d, Affine)
    assert tr3d.ndim == 3
    assert tr3d.linear_matrix.shape == (3, 3)
    assert tr3d.translate.shape == (3,)


def test_viewbox_to_world_matches_methods():
    viewer2d = _make_2d_viewer(center=(0, 4, 5), zoom=1.7)
    tr = viewer2d.viewbox_to_world
    rng = np.random.default_rng(0)
    for _ in range(5):
        cw = np.array([float(rng.uniform(0, 800)), float(rng.uniform(0, 600))])
        local = viewer2d.canvas_to_viewbox(tuple(cw), (0, 0))
        np.testing.assert_allclose(
            tr(local),
            viewer2d.canvas_to_world(tuple(cw))[list(viewer2d.dims.displayed)],
        )
        world = viewer2d.canvas_to_world(tuple(cw))
        np.testing.assert_allclose(
            tr.inverse(world[list(viewer2d.dims.displayed)]),
            viewer2d.canvas_to_viewbox(
                viewer2d.world_to_canvas(world), (0, 0)
            ),
        )

    viewer3d = _make_3d_viewer(
        center=(1, 2, 3), zoom=1.5, angles=(25, -30, 45)
    )
    tr3d = viewer3d.viewbox_to_world
    for _ in range(5):
        cw = (float(rng.uniform(0, 800)), float(rng.uniform(0, 600)))
        local = viewer3d.canvas_to_viewbox(cw, (0, 0))
        np.testing.assert_allclose(
            tr3d(np.array([0.0, *local])),
            viewer3d.canvas_to_world(cw),
        )
        world = viewer3d.canvas_to_world(cw)
        np.testing.assert_allclose(
            tr3d.inverse(world)[1:],
            viewer3d.canvas_to_viewbox(
                viewer3d.world_to_canvas(world), (0, 0)
            ),
        )
    # batched inverse also matches
    worlds = np.array(
        [
            viewer3d.canvas_to_world((100.0, 200.0)),
            viewer3d.canvas_to_world((300.0, 400.0)),
        ]
    )
    np.testing.assert_allclose(
        tr3d.inverse(worlds)[:, 1:],
        viewer3d.canvas_to_viewbox(viewer3d.world_to_canvas(worlds), (0, 0)),
    )


def test_viewbox_to_world_stays_in_sync():
    viewer = _make_2d_viewer(center=(0, 4, 5), zoom=1.0)
    tr = viewer.viewbox_to_world
    emissions = []
    tr.changed.connect(lambda: emissions.append(1))

    # same ndim: updated in place, so references stay valid
    viewer.scene.camera.zoom = 3.0
    assert viewer.viewbox_to_world is tr
    np.testing.assert_allclose(tr.linear_matrix, np.eye(2) / 3.0)
    world = np.array(viewer.scene.camera.center)
    np.testing.assert_allclose(
        viewer.world_to_canvas(world), tr.inverse(world[-2:])
    )
    assert len(emissions) > 0

    viewer.scene.camera.center = (0, 10, 20)
    assert viewer.viewbox_to_world is tr
    np.testing.assert_allclose(
        tr.translate, np.array([10, 20]) - np.array([400, 300]) / 3.0
    )

    viewer.canvas.size = (400, 300)
    assert viewer.viewbox_to_world is tr
    # world -> canvas still roundtrips after resize
    cw = (37.0, 123.0)
    np.testing.assert_allclose(
        viewer.world_to_canvas(viewer.canvas_to_world(cw)), cw
    )

    # ndisplay change replaces the object (dimensionality changes)
    viewer.dims.ndisplay = 3
    tr3d = viewer.viewbox_to_world
    assert tr3d is not tr
    assert tr3d.ndim == 3
    cw = (37.0, 123.0)
    np.testing.assert_allclose(
        viewer.world_to_canvas(viewer.canvas_to_world(cw)), cw
    )


def test_canvas_viewbox_roundtrip():
    viewer = _make_2d_viewer()
    viewer.add_image(np.ones((5, 5)))
    viewer.canvas.grid.enabled = True
    viewbox_size = np.array(viewer.canvas.viewbox_size(viewer.layers))
    # grid off -> identity
    viewer.canvas.grid.enabled = False
    np.testing.assert_allclose(
        viewer.canvas_to_viewbox((10.0, 20.0), (0, 0)), (10.0, 20.0)
    )
    viewer.canvas.grid.enabled = True
    # (row, col): col shifts x, row shifts y by full viewbox size (spacing=0)
    np.testing.assert_allclose(
        viewer.canvas_to_viewbox(viewbox_size / 2, (0, 0)), viewbox_size / 2
    )
    np.testing.assert_allclose(
        viewer.canvas_to_viewbox(
            viewbox_size / 2 + [0, viewbox_size[1]], (0, 1)
        ),
        viewbox_size / 2,
    )
    np.testing.assert_allclose(
        viewer.canvas_to_viewbox(
            viewbox_size / 2 + [viewbox_size[0], 0], (1, 0)
        ),
        viewbox_size / 2,
    )
    for vb in [(0, 0), (0, 1), (1, 0)]:
        local = (37.0, 123.0)
        global_pos = viewer.viewbox_to_canvas(local, vb)
        np.testing.assert_allclose(
            viewer.canvas_to_viewbox(tuple(global_pos), vb), local
        )
        # batched
        batch = np.array([local, local])
        np.testing.assert_allclose(
            viewer.canvas_to_viewbox(viewer.viewbox_to_canvas(batch, vb), vb),
            batch,
        )


def test_canvas_viewbox_with_spacing():
    viewer = _make_2d_viewer()
    viewer.add_image(np.ones((5, 5)))
    viewer.canvas.grid.enabled = True
    viewer.canvas.grid.spacing = 10
    viewbox_size = np.array(viewer.canvas.viewbox_size(viewer.layers))
    step = viewbox_size + 10
    # origin of (0, 1) is shifted by one full viewbox + spacing in x
    np.testing.assert_allclose(viewer._viewbox_origin((0, 0)), (0, 0))
    np.testing.assert_allclose(viewer._viewbox_origin((0, 1)), (0, step[1]))
    np.testing.assert_allclose(viewer._viewbox_origin((1, 0)), (step[0], 0))
    # camera center projects to the local center of every viewbox...
    world = np.array(viewer.scene.camera.center)
    for vb in [(0, 0), (0, 1)]:
        local = viewer.canvas_to_viewbox(viewer.world_to_canvas(world, vb), vb)
        np.testing.assert_allclose(local, viewbox_size / 2)
    # ...and roundtrips per viewbox
    for vb in [(0, 0), (0, 1)]:
        cw = (37.0, 123.0)
        global_pos = viewer.viewbox_to_canvas(
            viewer.canvas_to_viewbox(cw, vb), vb
        )
        np.testing.assert_allclose(global_pos, cw)


def test_3d_perspective_on_plane_roundtrip():
    # canvas_to_world returns a point on the camera-center plane (depth 0),
    # for which the perspective factor is 1, so canvas->world->canvas holds.
    viewer = _make_3d_viewer(angles=(20, 30, 40), perspective=45)
    rng = np.random.default_rng(0)
    for _ in range(10):
        cw = (float(rng.uniform(0, 800)), float(rng.uniform(0, 600)))
        world = viewer.canvas_to_world(cw)
        back = viewer.world_to_canvas(world)
        np.testing.assert_allclose(back, cw, rtol=1e-10, atol=1e-8)


def test_3d_perspective_off_plane_scaled():
    # Points off the center plane are scaled radially by dist/(dist-depth).
    viewer = _make_3d_viewer(
        center=(0, 0, 0), zoom=1.0, angles=(0, 0, 0), perspective=45
    )
    viewbox_size = np.array(viewer.canvas.viewbox_size(viewer.layers))
    vc = viewbox_size / 2
    h = float(viewbox_size[0])
    dist = (h / 1.0) / (2 * np.tan(np.radians(45) / 2))
    ortho = viewer.world_to_canvas(np.array([0, 10, 0]), viewbox=(0, 0))
    # with perspective=0 this is the orthographic projection
    viewer.scene.camera.perspective = 0
    expected_ortho = viewer.world_to_canvas(np.array([0, 10, 0]))
    np.testing.assert_allclose(ortho, expected_ortho)
    viewer.scene.camera.perspective = 45
    # depth 0 -> factor 1
    np.testing.assert_allclose(viewer.world_to_canvas(np.array([0, 0, 0])), vc)
    # depth = -10 (in front, towards viewer since view dir is -x?) check direction:
    # at zero angles, depth axis is dims[0]; offset [d, 0, 0] gives depth d.
    # dist - d in denominator; d<0 magnifies? verify against formula directly:
    for depth in (-50, -10, 10, 100):
        world = np.array([float(depth), 10.0, 0.0])
        got = viewer.world_to_canvas(world)
        factor = dist / (dist - depth)
        # recompute ortho for this y: y=10 maps to vc[0]+10
        ortho_y = vc + np.array([10.0, 0.0])
        np.testing.assert_allclose(got, vc + (ortho_y - vc) * factor)


def test_3d_perspective_behind_camera_nan():
    viewer = _make_3d_viewer(
        center=(0, 0, 0), zoom=1.0, angles=(0, 0, 0), perspective=45
    )
    viewbox_size = np.array(viewer.canvas.viewbox_size(viewer.layers))
    h = float(viewbox_size[0])
    dist = (h / 1.0) / (2 * np.tan(np.radians(45) / 2))
    # depth >= dist is behind the camera -> NaN
    behind = np.array([[dist + 10, 0, 0], [dist + 1000, 5, 5], [1e9, 0, 0]])
    out = viewer.world_to_canvas(behind)
    assert out.shape == (3, 2)
    assert np.all(np.isnan(out))
    # single behind point
    assert np.all(np.isnan(viewer.world_to_canvas(np.array([dist + 1, 0, 0]))))
    # just in front is finite and large
    front = viewer.world_to_canvas(np.array([dist - 1, 0, 0]))
    assert np.all(np.isfinite(front))


def test_3d_perspective_zero_is_ortho():
    viewer = _make_3d_viewer(angles=(15, 25, 35), perspective=0)
    rng = np.random.default_rng(3)
    worlds = rng.normal(size=(5, 3)) * 50
    ortho = viewer.world_to_canvas(worlds)
    viewer.scene.camera.perspective = 45
    persp = viewer.world_to_canvas(worlds)
    # they differ off-plane (sanity: perspective does something)
    assert not np.allclose(ortho, persp)
    viewer.scene.camera.perspective = 0
    np.testing.assert_allclose(viewer.world_to_canvas(worlds), ortho)


def test_batch_shapes():
    viewer2d = _make_2d_viewer()
    out = viewer2d.world_to_canvas(np.ones((4, 3)))
    assert out.shape == (4, 2)
    viewer3d = _make_3d_viewer()
    out = viewer3d.world_to_canvas(np.ones((4, 3)))
    assert out.shape == (4, 2)
    out = viewer3d.world_to_canvas(np.ones(3))
    assert out.shape == (2,)
