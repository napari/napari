import numpy as np

from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from ...utils.transforms import Affine


class InteractionBox(EventedModel):
    """Models a box that can be used to transform an object or a set of objects

    Parameters
    ----------
    points : list
        Nx2 array of points whose interaction box is to be found
    show : bool
        Bool indicating whether the box should be drawn
    show_handle : bool
        Bool indicating whether the full box with midpoints and rotation handle should be drawn.
        If False only the corners are drawn.
    show_vertices : bool
        Bool indicating whether vertices that enable interaction with the box should be drawn
    transform : napari.util.transforms.Affine
        Holds an affine transform that is modified by interaction with the vertices of the box
    selection_box_drag : list
        4x2 list of a box that is being drawn to select items. Gets updated during the mouse-drag
    selection_box_final : list
        4x2 list of a box that has been drawn to select item. Gets updated once the mouse drag is finished
    transform_start : napari.util.transforms.Affine
        The affine transformation of the box before a new mouse drag was started. Gets updated when a new drag begins.
    transform_drag : napari.util.transforms.Affine
        The affine transformation of the box during a mouse drag. Gets updated when mouse moves during drag.
    transform_final : napari.util.transforms.Affine
        The affine transformation of the box after a mouse drag. Gets updated when a mouse drag is finished.
    allow_new_selection: bool
        Bool indicating whether the interaction box allows creating a new selection by dragging outside an existing interaction_box.
    """

    points: Array[float, (-1, 2)] = None
    show: bool = False
    show_handle: bool = False
    show_vertices: bool = False
    selection_box_drag: Array[float, (4, 2)] = None
    selection_box_final: Array[float, (4, 2)] = None
    transform_start: Affine = Affine()
    transform_drag: Affine = Affine()
    transform_final: Affine = Affine()
    transform: Affine = Affine()
    allow_new_selection: bool = True
    selected_vertex: int = None

    @property
    def _box(self):
        box = self._create_box_from_points()
        if box is None:
            return None
        if self.transform:
            return self.transform(box)
        return box

    def _create_box_from_points(self):
        """Creates the axis aligned interaction box from the list of points"""
        if self.points is None or len(self.points) < 1:
            return None

        data = self.points

        min_val = np.array([data[:, 0].min(axis=0), data[:, 1].min(axis=0)])
        max_val = np.array([data[:, 0].max(axis=0), data[:, 1].max(axis=0)])

        tl = np.array([min_val[0], min_val[1]])
        tr = np.array([max_val[0], min_val[1]])
        br = np.array([max_val[0], max_val[1]])
        bl = np.array([min_val[0], max_val[1]])
        # If there is only one point avoid the corners overlapping in the singularity
        if len(self.points) == 1:
            tl += np.array([-0.5, -0.5])
            tr += np.array([0.5, -0.5])
            bl += np.array([0.5, 0.5])
            br += np.array([-0.5, 0.5])
        box = np.array(
            [
                tl,
                (tl + tr) / 2,
                tr,
                (tr + br) / 2,
                br,
                (br + bl) / 2,
                bl,
                (bl + tl) / 2,
                (tl + tr + br + bl) / 4,
            ]
        )
        return box
