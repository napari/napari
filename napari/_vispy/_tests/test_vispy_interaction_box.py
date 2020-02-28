from napari import Viewer
import numpy as np


def test_vispy_interacton_box(qtbot):

    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 15, 8))
    layer = viewer.add_image(data)
    layer._interaction_box.show = True
    layer._interaction_box.points = np.array([[0, 0], [100, 100]])

    assert (
        view._interaction_box_visual.marker_node._data['a_position'].shape[0]
        == 9
    )

    viewer.add_image(data)

    assert (
        view._interaction_box_visual.marker_node._data['a_position'].shape[0]
        == 1
    )

    viewer.active_layer = None

    assert (
        view._interaction_box_visual.marker_node._data['a_position'].shape[0]
        == 1
    )
