import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

from vispy.geometry.rect import Rect
from vispy.util.quaternion import Quaternion


def interpolate(states_dict):
    """Calculate interpolations for all states

    Returns
    -------
    interpolated: dict
        dictionary defining interpolated states. Each element is a list of length N
        frames. Keys are:
            'camera': list of N frames of camera states
            'ndisplay': array of legnth N frames indicating 2 or 3
            'sliders': array N frames x M sliders
            'vis': boolean array N frames x M layers
    """

    interpolated = {}
    interpolated["ndisplay"] = interpol_prop_zero(states_dict, "ndisplay")
    interpolated["vis"] = interpol_prop_zero(states_dict, "vis")
    interpolated["sliders"] = interpol_prop_lin(states_dict, "sliders").astype(
        np.uint8
    )
    cam2, cam3 = interpolate_camera(states_dict)

    # camera state is calculated for 2D and 3D camera for whole movie.
    # in the final list only add one camera per frame
    interpolated["camera"] = []
    for ind, x in enumerate(interpolated["ndisplay"]):
        if x == 2:
            if ind >= len(cam2):
                interpolated["camera"].append(cam2[-1])
            else:
                interpolated["camera"].append(cam2[ind])
        else:
            if ind >= len(cam3):
                interpolated["camera"].append(cam3[-1])
            else:
                interpolated["camera"].append(cam3[ind])

    return interpolated


def interpolate_camera(state_dict):
    """Create list of interpolated camera states

    Parameters
    -------
    states_dict : state dict as create by naparimovie.create_steps()

     Returns
    -------
    camera_states2D: list
        N frames list of 2D vispy camera states
    camera_states3D: list
        N frames list of 3D vispy camera states
    """

    # recover all frames and all frames with camera changes
    frames = [x["frame"] for x in state_dict]
    frames_cam = [[x["frame"], x["camera"]] for x in state_dict if x["camera"]]

    # Interpolate 2D camera
    camera_states2D = []
    frames2D = [x[0] for x in frames_cam if "rect" in x[1].keys()]
    if len(frames2D) > 0:

        # recover camera rectangle props
        all_rect = np.array(
            [
                [
                    x[1]["rect"].pos[0],
                    x[1]["rect"].pos[1],
                    x[1]["rect"].size[0],
                    x[1]["rect"].size[1],
                ]
                for x in frames_cam
                if "rect" in x[1].keys()
            ]
        )
        rect_interp = [
            Rect(*x)
            for x in interp1d(
                frames2D,
                all_rect,
                axis=0,
                bounds_error=False,
                fill_value=(all_rect[0, :], all_rect[-1, :]),
            )(frames)
        ]
        camera_states2D = [{"rect": x} for x in rect_interp]

    # Interpolate 3D camera
    camera_states3D = []
    frames3D = [x[0] for x in frames_cam if "_quaternion" in x[1].keys()]
    if len(frames3D) > 0:
        # recover rotation, translation, scale and rotation and interpolate
        all_rot = [
            x[1]["_quaternion"]
            for x in frames_cam
            if "_quaternion" in x[1].keys()
        ]
        all_trans = np.array(
            [
                x[1]["center"]
                for x in frames_cam
                if "_quaternion" in x[1].keys()
            ]
        )
        all_scale = np.array(
            [
                x[1]["scale_factor"]
                for x in frames_cam
                if "_quaternion" in x[1].keys()
            ]
        )

        rot_interp = [Quaternion(*x) for x in quat_interpol(frames3D, all_rot)]
        trans_interp = [
            tuple(x)
            for x in interp1d(
                frames3D,
                all_trans,
                axis=0,
                bounds_error=False,
                fill_value=(all_trans[0, :], all_trans[-1, :]),
            )(frames)
        ]
        scales_interp = np.interp(x=frames, xp=frames3D, fp=all_scale)

        camera_states3D = [
            {"scale_factor": x, "center": y, "fov": 0, "_quaternion": z}
            for (x, y, z) in zip(scales_interp, trans_interp, rot_interp)
        ]

    return camera_states2D, camera_states3D


def interpol_prop_zero(states_dict, prop):
    """For the property prop of the states_dict,
    interpolate missing frames between key-frames, by degree 0
    interpolation (replicate last state)

    Parameters
    -------
    states_dict : state dict as create by naparimovie.create_steps()
    prop : str
        property to interpolate

     Returns
    -------
    completed_values: array
        N frames x M property freatures
    """

    frames = [x["frame"] for x in states_dict if x[prop]]

    values = [x[prop] for x in states_dict if x[prop]]

    completed_values = np.concatenate(
        [
            [values[x] for i in range(frames[x], frames[x + 1])]
            for x in range(len(frames) - 1)
        ]
        + [[values[-1]]]
    )
    return completed_values


def interpol_prop_lin(states_dict, prop):
    """For the property prop of the states_dict,
    interpolate missing frames between key-frames, by linear interpolation

    Parameters
    -------
    states_dict : state dict as create by naparimovie.create_steps()
    prop : str
        property to interpolate

     Returns
    -------
    value_interp: array
        N frames x M property freatures
    """

    frames = [x["frame"] for x in states_dict]
    frames_values = [x["frame"] for x in states_dict if x[prop]]
    values = [x[prop] for x in states_dict if x[prop]]

    value_interp = interp1d(frames_values, np.stack(values), axis=0)(frames)

    return value_interp


def quat_interpol(frames_rot, rot_states):
    """Interpolate camera rotation state using
    quaternions

    Parameters
    -------
    frames_rot : list
        list of frames with camera rotation changes
    rot_states : list
        list of vispy quaternions

     Returns
    -------
    all_states: array
        N frames x 4 array with each line a quaternion (w,x,y,z)
    """

    if frames_rot[0] != 0:
        frames_rot = [0] + frames_rot
        rot_states = [rot_states[0]] + rot_states

    all_states = []
    for i in range(len(frames_rot) - 1):

        q = R.from_quat(
            [
                [
                    rot_states[i].w,
                    rot_states[i].x,
                    rot_states[i].y,
                    rot_states[i].z,
                ],
                [
                    rot_states[i + 1].w,
                    rot_states[i + 1].x,
                    rot_states[i + 1].y,
                    rot_states[i + 1].z,
                ],
            ]
        )

        num_frames = frames_rot[i + 1] - frames_rot[i] - 1
        slerp = Slerp([0, num_frames + 1], q)
        # do not repeat the point connecting interpolation sequences
        if i == len(frames_rot) - 2:
            times = np.arange(num_frames + 2)
        else:
            times = np.arange(num_frames + 1)
        interp_rots = slerp(times)
        all_states.append(interp_rots.as_quat())

    all_states = np.concatenate(all_states)

    return all_states
