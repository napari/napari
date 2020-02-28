import imageio
import copy
import skimage.transform
import skimage.io
import numpy as np
from pathlib import Path

from . import util


class Animation:
    def __init__(self, viewer=None, inter_steps=15):

        """Standard __init__ method.

        Parameters
        ----------
        viewer : viewer
            napari viewer
        inter_steps: int
            number of steps to interpolate between key frames

        Attributes
        ----------
        key_frames : list
            list of dictionary defining viewer states. Dictionaries have keys:
                'frame': int, frame
                'ndisplay': int, view type (2,3)
                'camera': camera state
                'vis': list of booleans, visibility of layers
                'sliders': list, slider positions

        interpolated_states: dict
            dictionary defining interpolated states. Each element is a list of length N
            frames. Keys are:
                'ndisplay': int, view type (2,3)
                'camera': camera state
                'vis': list of booleans, visibility of layers
                'sliders': list, slider positions

        states_dict : list
            list of dictionaries defining viewer states for each frame. Same keys as key_frames

        current_frame : int
            currently shown key frame
        implot : matplotlib Ax object
            reference to matplotlib image used for movie returned by imshow
        anim : matplotlib FuncAnimation object
            reference to animation object
        """

        if viewer is None:
            raise TypeError(
                "You need to pass a napari viewer for the viewer argument"
            )
        else:
            self.viewer = viewer

        self.key_frames = []
        self.inter_steps = inter_steps

        self.current_frame = -1
        self.current_interpolframe = 0

        # establish key bindings
        self.add_callback()

    def finish_movie(self):
        self.release_callbacks()

    def add_callback(self):
        """Bind keys"""

        self.viewer.bind_key("f", self.capture_keyframe_callback)
        self.viewer.bind_key("r", self.replace_keyframe_callback)
        self.viewer.bind_key("d", self.delete_keyframe_callback)

        self.viewer.bind_key("a", self.key_adv_frame)
        self.viewer.bind_key("b", self.key_back_frame)

        self.viewer.bind_key("w", self.key_interpolframe)

    def release_callbacks(self):
        """Release keys"""

        self.viewer.bind_key("f", None)
        self.viewer.bind_key("r", None)
        self.viewer.bind_key("d", None)

        self.viewer.bind_key("a", None)
        self.viewer.bind_key("b", None)

        self.viewer.bind_key("w", None)

    def get_new_state(self):
        """Capture current viewer state

        Returns
        -------
        new_state : dict
            description of state
        """

        new_state = {
            "ndisplay": self.viewer.dims.ndisplay,
            "frame": self.current_frame,
            "camera": copy.deepcopy(
                self.viewer.window.qt_viewer.view.camera.get_state()
            ),
            "vis": [x.visible for x in self.viewer.layers],
            "sliders": self.viewer.dims.point,
        }

        return new_state

    def capture_keyframe_callback(self, viewer):
        """Record current key-frame"""

        new_state = self.get_new_state()
        new_state["frame"] += 1
        self.key_frames.insert(self.current_frame + 1, new_state)
        self.current_frame += 1

    def replace_keyframe_callback(self, viewer):
        """Replace current key-frame with new view"""

        new_state = self.get_new_state()
        self.key_frames[self.current_frame] = new_state

        self.create_steps()

    def delete_keyframe_callback(self, viewer):
        """Delete current key-frame"""

        self.key_frames.pop(self.current_frame)

        self.current_frame = (self.current_frame - 1) % len(self.key_frames)
        self.set_to_keyframe(self.current_frame)
        self.create_steps()

    def key_adv_frame(self, viewer):
        """Go forwards in key-frame list"""

        new_frame = (self.current_frame + 1) % len(self.key_frames)
        self.set_to_keyframe(new_frame)

    def key_back_frame(self, viewer):
        """Go backwards in key-frame list"""

        new_frame = (self.current_frame - 1) % len(self.key_frames)
        self.set_to_keyframe(new_frame)

    def set_to_keyframe(self, frame):
        """Set the viewer to a given key-frame

        Parameters
        -------
        frame : int
            key-frame to visualize
        """

        self.current_frame = frame

        for i in range(len(self.key_frames[frame]["sliders"])):
            self.viewer.dims.set_point(i, self.key_frames[frame]["sliders"][i])

        # set visibility of layers
        for j in range(len(self.viewer.layers)):
            self.viewer.layers[j].visible = self.key_frames[frame]["vis"][j]

        # update state
        self.viewer.dims.ndisplay = self.key_frames[frame]["ndisplay"]
        self.viewer.window.qt_viewer.view.camera.set_state(
            self.key_frames[frame]["camera"]
        )
        self.viewer.window.qt_viewer.view.camera.view_changed()

    def create_state_dict(self):
        """Create list of state dictionaries. For key-frames selected interactively,
        add self.inter_steps emtpy frames between key-frames. For key-frames from scripts,
        the number of empty frames ot add between each key-frame is already set in self.inter_steps.
        """

        if type(self.inter_steps) is not list:
            inter_steps = len(self.key_frames) * [self.inter_steps]
        else:
            inter_steps = self.inter_steps

        empty = {
            "ndisplay": [],
            "frame": [],
            "camera": [],
            "vis": [],
            "sliders": [],
        }
        states_dict = []
        for ind, x in enumerate(self.key_frames):
            states_dict.append(x)
            # do not add frames after last key-frame
            if ind < len(self.key_frames) - 1:
                # do not add frames when switching camera
                if (
                    self.key_frames[ind]['ndisplay']
                    == self.key_frames[ind + 1]['ndisplay']
                ):
                    for y in range(inter_steps[ind]):
                        states_dict.append(copy.deepcopy(empty))
        for ind, x in enumerate(states_dict):
            x["frame"] = ind
        self.states_dict = states_dict

    def create_steps(self):
        """Interpolate states between key-frames"""

        self.create_state_dict()
        self.interpolated_states = util.interpolate(self.states_dict)

    def key_interpolframe(self, viewer):
        """Progress through interpolated frames"""

        self.create_steps()

        new_frame = (self.current_interpolframe + 1) % len(self.states_dict)
        self.update_viewer_from_state(new_frame)
        self.current_interpolframe = new_frame

    def update_viewer_from_state(self, frame):
        """Set the viewer to a given interpolated frame

        Parameters
        -------
        frame : int
            frame to visualize
        """

        # set view type 2D/3D and camera state
        self.viewer.dims.ndisplay = self.interpolated_states["ndisplay"][frame]
        self.viewer.window.qt_viewer.view.camera.set_state(
            self.interpolated_states["camera"][frame]
        )

        # assign interpolated visibility state
        for j in range(len(self.viewer.layers)):
            self.viewer.layers[j].visible = self.interpolated_states["vis"][
                frame
            ][j]

        # adjust slider positions
        for i in range(self.interpolated_states["sliders"].shape[1]):
            self.viewer.dims.set_point(
                i, self.interpolated_states["sliders"][frame][i]
            )

        # update view
        self.viewer.window.qt_viewer.view.camera.view_changed()

    def frame_generator(self, frame=None, with_viewer=False):
        """Generator of frames of the animation

        Parameters
        -------
        frame : int
            Specific frame to return.
        with_viewer : bool
            If True includes the napari viewer, otherwise just includes the
            canvas.

        """

        # create states
        self.create_steps()

        # capture SceneCanvas size of frist frame to set size of next ones
        self.update_viewer_from_state(0)
        frame_size = self.viewer.window.qt_viewer.canvas.size

        # return specific frame
        if frame is not None:
            self.update_viewer_from_state(frame)
            image = self.viewer.screenshot(with_viewer=with_viewer)
            while True:
                yield image
        # return all frames as a generator
        else:
            for i in range(len(self.states_dict)):
                self.update_viewer_from_state(i)
                if not with_viewer:
                    self.viewer.window.qt_viewer.canvas.size = frame_size
                yield self.viewer.screenshot(with_viewer=with_viewer)

    def animate(
        self,
        name="movie.mp4",
        fps=20,
        quality=5,
        format=None,
        with_viewer=False,
        scale_factor=None,
    ):
        """Create a movie based on key-frames

        Parameters
        -------
        name : str
            name to use for saving the movie (can also be a path)
            should be either .mp4 or .gif. If no extension is provided,
            images are saved as a folder of PNGs
        fps : int
            frames per second
        quality: float
            number from 1 (lowest quality) to 9
            only applies to mp4
        format: str
            The format to use to write the file. By default imageio selects the appropriate for you based on the filename.
        with_viewer : bool
            If True includes the napari viewer, otherwise just includes the
            canvas.
        scale_factor : float
            Rescaling factor for the image size. Only used without
            viewer (with_viewer = False).
        """

        # create a frame generator
        frame_gen = self.frame_generator(with_viewer=with_viewer)

        # create path object
        path = Path(name)

        # if path has no extension, save as fold of PNG
        save_as_folder = False
        if path.suffix == "":
            save_as_folder = True

        # try to create an ffmpeg writer. If not installed default to folder creation
        if not save_as_folder:
            try:
                # create imageio writer and add all frames
                if quality is not None:
                    writer = imageio.get_writer(
                        name, fps=fps, quality=quality, format=format,
                    )
                else:
                    writer = imageio.get_writer(name, fps=fps, format=format)
            except ImportError as err:
                print(err)
                print('Your movie will be saved as a series of PNG files.')
                save_as_folder = True

        # if movie is saved as series of PNG, create a folder
        if save_as_folder:
            folder_path = path.absolute()
            folder_path = path.parent.joinpath(path.stem)
            folder_path.mkdir(exist_ok=True)

        # save frames
        for ind, frame in enumerate(frame_gen):
            if scale_factor is not None:
                frame = skimage.transform.rescale(
                    frame, scale_factor, multichannel=True, preserve_range=True
                )
                frame = frame.astype(np.uint8)
            if not save_as_folder:
                writer.append_data(frame)
            else:
                skimage.io.imsave(
                    folder_path.joinpath(path.stem + '_' + str(ind) + '.png'),
                    frame,
                )

        if not save_as_folder:
            writer.close()
