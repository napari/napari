import numpy as np

from ..utils.events import EventedModel
from ..utils.events.custom_types import Array
from ..utils.transforms import Transform
from ._interaction_box_constants import Box


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

    """

    points: Array[float, (-1, 2)] = None
    show: bool = False
    show_handle: bool = False
    show_vertices: bool = False
    selection_box_drag: Array[float, (4, 2)] = None
    selection_box_final: Array[float, (4, 2)] = None
    transform_start: Transform = None
    transform_drag: Transform = None
    transform_final: Transform = None
    transform: Transform = None
    angle: float = 0
    rotation_handle_length = 20
    allow_new_selection: bool = True

    def __init__(self, points=None, show=False, show_handle=False):

        super().__init__(points=points, show=show, show_handle=show_handle)

    @property
    def _box(self):
        box = self._create_box_from_points()
        if box is None:
            return None
        if self.transform:
            return self.transform(box)
        else:
            return box

    def _add_rotation_handle(self, box):
        """Adds the rotation handle to the box"""

        if box is not None:
            rot = box[Box.TOP_CENTER]
            length_box = np.linalg.norm(
                box[Box.BOTTOM_LEFT] - box[Box.TOP_LEFT]
            )
            if length_box > 0:
                r = self.rotation_handle_length
                rot = (
                    rot
                    - r
                    * (box[Box.BOTTOM_LEFT] - box[Box.TOP_LEFT])
                    / length_box
                )
            box = np.append(box, [rot], axis=0)

        return box

    def _create_box_from_points(self):
        """Creates the axis aligned interaction box from the list of points"""
        if self.points is None or len(self.points) < 1:
            return None

        if len(self.points) == 1:
            point = self.points[0]
            tl = point + np.array([-0.5, -0.5])
            tr = point + np.array([0.5, -0.5])
            bl = point + np.array([0.5, 0.5])
            br = point + np.array([-0.5, 0.5])
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
            return self._add_rotation_handle(box)

        data = self.points

        min_val = np.array([data[:, 0].min(axis=0), data[:, 1].min(axis=0)])
        max_val = np.array([data[:, 0].max(axis=0), data[:, 1].max(axis=0)])
        tl = np.array([min_val[0], min_val[1]])
        tr = np.array([max_val[0], min_val[1]])
        br = np.array([max_val[0], max_val[1]])
        bl = np.array([min_val[0], max_val[1]])
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
        return self._add_rotation_handle(box)
