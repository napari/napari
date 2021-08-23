import numpy as np
from vispy.app.canvas import MouseEvent

import napari


def test_closed_viewer_ok():
    v = napari.Viewer()
    v.close()
    v.add_points()
    v.add_image(np.random.rand(4, 4))
    v.show()
    v.window.qt_viewer.on_mouse_move(MouseEvent('move'))
    v._on_cursor_position_change(None)
    v.close()
    v.show()


def test_reuse():
    v = napari.Viewer()
    v.add_points()
    v.add_image(np.random.rand(4, 4))
    v.close()
    v.show()


def test_add_then_show():
    v = napari.Viewer(show=False)
    v.add_points()
    v.add_image(np.random.rand(4, 4))
    v.show()


def test_qt_disconnected(make_napari_viewer):
    v = napari.Viewer()
    v.add_points()
    v.add_image(np.random.rand(8, 8))
    v.add_labels(np.random.rand(8, 8).astype(int))
    v.close()
