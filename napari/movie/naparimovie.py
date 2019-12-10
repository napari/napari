import numpy as np
import matplotlib.pyplot as plt
import napari
from pyquaternion import Quaternion
from matplotlib.animation import FuncAnimation
import imageio
import copy
import vispy.geometry

from . import util

class Movie:
    def __init__(self, myviewer=None, inter_steps=15):

        """Standard __init__ method.
        
        Parameters
        ----------
        myviewer : napari viewer
            napari viewer
        inter_steps: int
            number of steps to interpolate between key frames 
        
        Attributes
        ----------
        key_frames : list
            list of dictionary defining napari viewer states. Dictionaries have keys:
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
            list of dictionaries defining napari viewer states for each frame. Same keys as key_frames
        
        current_frame : int
            currently shown key frame
        implot : matplotlib Ax object
            reference to matplotlib image used for movie returned by imshow
        anim : matplotlib FuncAnimation object
            reference to animation object
            
        """

        if myviewer is None:
            raise TypeError(
                "You need to pass a napari viewer for the myviewer argument"
            )
        else:
            self.myviewer = myviewer

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

        self.myviewer.bind_key("f", self.capture_keyframe_callback)
        self.myviewer.bind_key("r", self.replace_keyframe_callback)
        self.myviewer.bind_key("d", self.delete_keyframe_callback)

        self.myviewer.bind_key("a", self.key_adv_frame)
        self.myviewer.bind_key("b", self.key_back_frame)

        self.myviewer.bind_key("w", self.key_interpolframe)
        
    def release_callbacks(self):
        """Release keys"""
        
        self.myviewer.bind_key("f", None)
        self.myviewer.bind_key("r", None)
        self.myviewer.bind_key("d", None)

        self.myviewer.bind_key("a", None)
        self.myviewer.bind_key("b", None)

        self.myviewer.bind_key("w", None)
        
    
    def get_new_state(self):
        """Capture current napari state
        
        Returns
        -------
        new_state : dict
            description of state
        """

        current_state = copy.deepcopy(
            self.myviewer.window.qt_viewer.view.camera.get_state()
        )
        time = self.myviewer.dims.point[0] if len(self.myviewer.dims.point) == 4 else []
        new_state = {
            "ndisplay": self.myviewer.dims.ndisplay,
            "frame": self.current_frame,
            "camera": copy.deepcopy(self.myviewer.window.qt_viewer.view.camera.get_state()),
            "vis": [x.visible for x in self.myviewer.layers],
            "sliders": self.myviewer.dims.point,
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
        """Set the napari viewer to a given key-frame
        
        Parameters
        -------
        frame : int
            key-frame to visualize
        """

        self.current_frame = frame

        for i in range(len(self.key_frames[frame]["sliders"])):
            self.myviewer.dims.set_point(i, self.key_frames[frame]["sliders"][i])
        
        # set visibility of layers
        for j in range(len(self.myviewer.layers)):
            #if self.key_frames[frame]["vis"]:
            self.myviewer.layers[j].visible = self.key_frames[frame]["vis"][j]
        
        # update state
        self.myviewer.dims.ndisplay = self.key_frames[frame]["ndisplay"]
        self.myviewer.window.qt_viewer.view.camera.set_state(self.key_frames[frame]["camera"])
        self.myviewer.window.qt_viewer.view.camera.view_changed()

        
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
            #do not add frames after last key-frame
            if ind < len(self.key_frames) - 1:
                #do not add frames when switching camera
                if self.key_frames[ind]['ndisplay'] == self.key_frames[ind+1]['ndisplay']:
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
        self.update_napari_state(new_frame)
        self.current_interpolframe = new_frame

    def collect_images(self):
        """Collect images corresponding to all interpolated states
        
        Returns
        -------
        image_stack : 3D numpy
            stack of all snapshots
        """

        images = []
        self.create_steps()
        for i in range(len(self.states_dict)):

            self.update_napari_state(i)
            images.append(self.myviewer.screenshot())

        image_stack = np.stack(images, axis=0)
        return image_stack

    def update_napari_state(self, frame):
        """Set the napari viewer to a given interpolated frame
        
        Parameters
        -------
        frame : int
            frame to visualize
        """

        #set view type 2D/3D and camera state
        self.myviewer.dims.ndisplay = self.interpolated_states["ndisplay"][frame]
        self.myviewer.window.qt_viewer.view.camera.set_state(self.interpolated_states["camera"][frame])
        
        #assign interpolated visibility state
        for j in range(len(self.myviewer.layers)):
            self.myviewer.layers[j].visible = self.interpolated_states["vis"][frame][j]

        #adjust slider positions
        for i in range(self.interpolated_states["sliders"].shape[1]):
            self.myviewer.dims.set_point(i, self.interpolated_states["sliders"][frame][i])
            
        #update view
        self.myviewer.window.qt_viewer.view.camera.view_changed()

        
    def create_movie_frame(self):
        """Create the matplotlib figure, and image object hosting all snapshots"""

        newim = self.myviewer.screenshot()
        sizes = newim.shape
        height = float(sizes[0])
        width = float(sizes[1])

        factor = 3
        fig = plt.figure()
        fig.set_size_inches(factor * width / height, factor, forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        self.fig = fig
        self.ax = ax

        self.implot = plt.imshow(newim, animated=True)

    def movie_init(self):
        """init function for matplotlib FuncAnimation"""

        newim = self.myviewer.screenshot()
        self.implot.set_data(newim)
        return self.implot

    def update(self, frame):
        """Update function matplotlib FuncAnimation 
        
        Parameters
        -------
        frame : int
            frame to visualize
        """

        self.update_napari_state(frame)
        newim = self.myviewer.screenshot()
        self.implot.set_data(newim)
        return self.implot

    def make_movie(self, name="movie.mp4", resolution=600, fps=20):
        """Create a movie based on key-frames selected in napari
        
        Parameters
        -------
        name : str
            name to use for saving the movie (can also be a path)
        resolution: int
            resolution in dpi to save the movie
        fps : int
            frames per second
        """

        # creat all states
        self.create_steps()
        # create movie frame
        self.create_movie_frame()
        # animate
        self.anim = FuncAnimation(
            self.fig,
            self.update,
            frames=np.arange(len(self.states_dict)),
            init_func=self.movie_init,
            blit=False,
        )
        plt.show()

        self.anim.save(name, dpi=resolution, fps=fps)

    def make_gif(self, name="movie.gif"):
        """Create a gif based on key-frames selected in napari
        
        Parameters
        -------
        name : str
            name to use for saving the movie (can also be a path)
        """

        # create the image stack with all snapshots
        stack = self.collect_images()

        imageio.mimsave(name, [stack[i, :, :, :] for i in range(stack.shape[0])])