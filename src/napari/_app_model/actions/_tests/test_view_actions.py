import numpy as np

from napari._app_model.actions._view import _toggle_canvas_ndim
from napari.components import ViewerModel


def test_view_menu_ndisplay_toggle_respects_navigation_lock():
    """The View menu's 2D/3D toggle (a separate implementation from the
    keybinding action) must honor the navigation lock too, so it cannot change
    the displayed axes while a slice-keyed operation is locked."""
    viewer = ViewerModel()
    viewer.add_image(np.zeros((4, 5, 6)))
    assert viewer.dims.ndisplay == 2

    token = object()
    viewer.dims.lock_navigation(token)
    _toggle_canvas_ndim(viewer)
    assert viewer.dims.ndisplay == 2  # blocked while locked

    viewer.dims.unlock_navigation(token)
    _toggle_canvas_ndim(viewer)
    assert viewer.dims.ndisplay == 3  # allowed once unlocked
