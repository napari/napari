import numpy as np

from napari._vispy.overlays.direction_labels import (
    VispyDirectionLabelsOverlay,
)
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel

# A standard axial DICOM-LPS frame: axis 0 (depth) I/S, axis 1 A/P, axis 2 R/L.
LPS_AXIAL = [('I', 'S'), ('A', 'P'), ('R', 'L')]


def _make_view(labels=LPS_AXIAL, ndim=3, ndisplay=2):
    viewer = ViewerModel()
    viewer.dims.ndim = ndim
    viewer.dims.ndisplay = ndisplay
    viewer.direction_labels.labels = labels
    view = VispyDirectionLabelsOverlay(
        viewer=viewer, overlay=viewer.direction_labels, font_info=FontInfo()
    )
    return viewer, view


def _shown(view):
    """Map edge -> label for the currently visible edge texts."""
    return {
        edge: text.text for edge, text in view._texts.items() if text.visible
    }


def test_default_orientation_places_each_label_on_its_edge():
    _viewer, view = _make_view()
    assert _shown(view) == {
        'top': 'A',
        'bottom': 'P',
        'left': 'R',
        'right': 'L',
    }


def test_vertical_flip_swaps_top_and_bottom():
    viewer, view = _make_view()
    viewer.camera.orientation = ('away', 'up', 'right')
    assert _shown(view) == {
        'top': 'P',
        'bottom': 'A',
        'left': 'R',
        'right': 'L',
    }


def test_horizontal_flip_swaps_left_and_right():
    viewer, view = _make_view()
    viewer.camera.orientation = ('away', 'down', 'left')
    assert _shown(view) == {
        'top': 'A',
        'bottom': 'P',
        'left': 'L',
        'right': 'R',
    }


def test_transposed_order_reassigns_edges():
    viewer, view = _make_view()
    viewer.dims.order = (
        0,
        2,
        1,
    )  # displayed == (2, 1): axis2 vert, axis1 horz
    assert _shown(view) == {
        'top': 'R',
        'bottom': 'L',
        'left': 'A',
        'right': 'P',
    }


def test_3d_hides_all_labels():
    viewer, view = _make_view()
    viewer.dims.ndisplay = 3
    assert _shown(view) == {}


def test_empty_labels_hides_all():
    _viewer, view = _make_view(labels=[])
    assert _shown(view) == {}


def test_labels_survive_ndim_growth_via_reconcile():
    # Labels set for a 2D world (a suffix frame) stay correct when a new leading
    # axis appears (as when a higher-dim layer is added).
    viewer, view = _make_view(labels=[('A', 'P'), ('R', 'L')], ndim=2)
    assert _shown(view) == {
        'top': 'A',
        'bottom': 'P',
        'left': 'R',
        'right': 'L',
    }
    viewer.dims.ndim = 3  # prepended axis is unlabeled; A/P, R/L shift right
    assert _shown(view) == {
        'top': 'A',
        'bottom': 'P',
        'left': 'R',
        'right': 'L',
    }


def test_partial_pair_labels_only_present_edges():
    _viewer, view = _make_view(labels=[None, (None, 'P'), ('R', None)])
    assert _shown(view) == {'bottom': 'P', 'left': 'R'}


def test_close_disconnects_model_events():
    # A closed (but retained) visual must not keep reacting to model changes.
    viewer, view = _make_view()
    before = _shown(view)
    view.close()
    viewer.camera.orientation = ('away', 'up', 'left')
    viewer.dims.ndisplay = 3
    viewer.direction_labels.labels = []
    assert _shown(view) == before


def _assert_all_within_canvas(view, width, height):
    """Every visible label's glyph box lies fully inside the canvas."""
    for edge, text in view._texts.items():
        if not text.visible:
            continue
        x, y = float(text.pos[0][0]), float(text.pos[0][1])
        tw, th = text.get_width_height()
        assert x - tw / 2 >= 0, f'{edge} clipped at left'
        assert x + tw / 2 <= width, f'{edge} clipped at right'
        assert y - th / 2 >= 0, f'{edge} clipped at top'
        assert y + th / 2 <= height, f'{edge} clipped at bottom'


def _viewer_view(qt_viewer):
    viewer = qt_viewer.viewer
    viewer.dims.ndim = 3
    viewer.direction_labels.labels = LPS_AXIAL
    viewer.direction_labels.visible = True
    return viewer, qt_viewer.canvas._overlay_to_visual[
        viewer.direction_labels
    ][0]


def test_resize_updates_positions_on_real_canvas(qt_viewer):
    # B1 regression: the node is parented in the base __init__, before our
    # canvas_change hook, so the overlay must connect the current canvas's
    # resize explicitly at construction. Without that, _resize_canvas is None
    # and real resizes never reposition the labels (the truncation bug).
    _viewer, view = _viewer_view(qt_viewer)
    assert view._resize_canvas is qt_viewer.canvas._scene_canvas

    # Emit the actual resize event (not a direct handler call), so this also
    # covers that our handler is wired to the canvas resize. The handler reads
    # the current canvas size, not the stale viewer._canvas_size.
    def _resize(h, w):
        qt_viewer.canvas.size = (h, w)
        sc = view._resize_canvas
        sc.events.resize(size=sc.size)

    _resize(600, 400)
    _assert_all_within_canvas(view, 400, 600)
    assert view._texts['bottom'].pos[0][1] > view._texts['top'].pos[0][1]

    _resize(300, 900)
    _assert_all_within_canvas(view, 900, 300)


def test_default_color_updates_on_theme_change(qt_viewer):
    # With color=None the text color contrasts the canvas, so it must refresh
    # when the theme changes (box=False makes the base theme handler a no-op).
    viewer, view = _viewer_view(qt_viewer)

    viewer.theme = 'light'
    light = np.array(view._texts['top'].color.rgba)
    viewer.theme = 'dark'
    dark = np.array(view._texts['top'].color.rgba)
    assert not np.allclose(light, dark)
