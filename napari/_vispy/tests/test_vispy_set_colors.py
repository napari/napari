import pytest
import numpy as np

from napari import Viewer
from vispy.color import ColorArray
from colors_data import *  # noqa: F403


@pytest.fixture(scope="function")
def setup_viewer(qtbot):
    """Instatiates and removes an instance of a Viewer().
    The use of 'yield' means that when the fixture is called, all
    code up to it will be run, and when the testing function is
    done, the code following the yield will be run.
    """
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)
    yield viewer, view
    viewer.window.close()


@pytest.mark.parametrize("points", all_one_points)
@pytest.mark.parametrize("colors", single_color_options)
def test_oned_points(setup_viewer, points, colors):
    if type(colors) is np.ndarray:
        colors = colors.ravel()
    viewer, view = setup_viewer
    viewer.add_points(points, face_color=colors, edge_color=colors)
    try:
        true_color = colors.rgba
    except AttributeError:
        true_color = ColorArray(colors).rgba

    assert true_color == viewer.layers[0].edge_colors
    assert true_color == viewer.layers[0].face_colors


@pytest.mark.parametrize("points", all_two_points)
@pytest.mark.parametrize("colors", two_color_options)
def test_twod_points(setup_viewer, points, colors):
    # If colors is an array then pytest's parametrize ruins the
    # dimensionality of it.
    if type(colors) is np.ndarray:
        colors = colors.ravel()
    viewer, view = setup_viewer
    viewer.add_points(points, face_color=colors, edge_color=colors)
    try:
        true_color = colors.rgba
    except AttributeError:
        true_color = ColorArray(colors).rgba

    assert true_color == viewer.layers[0].edge_colors
    assert true_color == viewer.layers[0].face_colors
