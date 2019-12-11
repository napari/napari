import numpy as np
from vispy.util.quaternion import Quaternion
from vispy.geometry.rect import Rect


from ..util import (
    interpolate,
    interpolate_camera,
    interpol_prop_zero,
    interpol_prop_lin,
    quat_interpol,
)


# create a mock state
frames = [0, 1, 2, 3, 4, 5]
ndisplay = [3, [], 3, 2, [], 2]
vis = [[True, True], [], [], [False, True], [], [True, False]]
sliders = [[0, 0], [], [], [0, 5], [], [5, 0]]
rot1 = Quaternion.create_from_axis_angle(0, 1, 0, 0)
rot2 = Quaternion.create_from_axis_angle(np.pi / 4, 1, 0, 0)
rect1 = Rect(0, 0, 100, 50)
rect2 = Rect(10, 20, 110, 30)
camera = [
    {"scale_factor": 1, "center": (0, 0), "fov": 0, "_quaternion": rot1},
    [],
    {"scale_factor": 2, "center": (20, -10), "fov": 0, "_quaternion": rot2},
    {'rect': rect1},
    [],
    {'rect': rect2},
]

states = [
    {
        'frame': frames[x],
        'ndisplay': ndisplay[x],
        'vis': vis[x],
        'sliders': sliders[x],
        'camera': camera[x],
    }
    for x in range(len(frames))
]

# create target values
interpol_ndisplay_true = np.array([3, 3, 3, 2, 2, 2])
interpol_vis_true = np.array(
    [
        [True, True],
        [True, True],
        [True, True],
        [False, True],
        [False, True],
        [True, False],
    ]
)
interpol_sliders_true = np.array(
    [[0, 0], [0, 1], [0, 3], [0, 5], [2, 2], [5, 0]]
)


def test_interpol_prop_zero():

    interpol_vis = interpol_prop_zero(states, 'vis')
    interpol_ndisplay = interpol_prop_zero(states, 'ndisplay')

    assert np.all(interpol_vis == interpol_vis_true)
    assert np.all(interpol_ndisplay == interpol_ndisplay_true)


def test_interpol_prop_lin():

    interpol_sliders = interpol_prop_lin(states, 'sliders').astype(np.uint8)
    assert np.all(interpol_sliders == interpol_sliders_true)


def test_quat_interpol():

    frames_rot = [0, 2]
    interpol_rotation = quat_interpol(frames_rot, [rot1, rot2])

    np.testing.assert_almost_equal(
        2 * np.arccos(interpol_rotation[1, 0]), np.pi / 8
    )


def test_interpolate_camera():
    cam2, cam3 = interpolate_camera(states)

    rect = np.array(
        [
            [
                x["rect"].left,
                x["rect"].bottom,
                x["rect"].width,
                x["rect"].height,
            ]
            for x in cam2
        ]
    )
    scale = [x["scale_factor"] for x in cam3]
    center = [x["center"] for x in cam3]

    rect_true = np.array(
        [
            [0.0, 0.0, 100.0, 50.0],
            [0.0, 0.0, 100.0, 50.0],
            [0.0, 0.0, 100.0, 50.0],
            [0.0, 0.0, 100.0, 50.0],
            [5.0, 10.0, 105.0, 40.0],
            [10.0, 20.0, 110.0, 30.0],
        ]
    )
    scale_true = [1.0, 1.5, 2.0]
    center_true = [(0.0, 0.0), (10.0, -5.0), (20.0, -10.0)]

    assert np.all(rect == rect_true)
    assert np.all(scale == scale_true)
    assert type(center[0]) is tuple
    assert np.all(center == center_true)
    np.testing.assert_almost_equal(
        cam3[1]["_quaternion"].get_axis_angle()[0], np.pi / 8
    )


def test_interpolate():

    interp_state = interpolate(states)

    assert np.all(interp_state['vis'] == interpol_vis_true)
    assert np.all(interp_state['ndisplay'] == interpol_ndisplay_true)
    assert np.all(interp_state['sliders'] == interpol_sliders_true)
