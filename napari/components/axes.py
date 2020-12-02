import numpy as np

from ..utils.colormaps.standardize_color import transform_single_color
from ..utils.events.dataclass import Property, evented_dataclass


@evented_dataclass
class Axes:
    """Axes indicating world coordinate origin and orientation.

    Attributes
    ----------
    visible : bool
        If axes are visible or not.
    labels : bool
        If axes labels are visible or not. Not the actual
        axes labels are stored in `viewer.dims.axes_labels`.
    colored : bool
        If axes are colored or not. If colored then default
        coloring is x=cyan, y=yellow, z=magenta. If not
        colored than axes are the color opposite of
        the canvas background.
    dashed : bool
        If axes are dashed or not. If not dashed then
        all the axes are solid. If dashed then x=solid,
        y=dashed, z=dotted.
    arrows : bool
        If axes have arrowheads or not.
    background_color : np.ndarray
        Background color of canvas. If axes are not colored
        then they have the color opposite of this color.
    """

    visible: bool = False
    labels: bool = True
    colored: bool = True
    dashed: bool = False
    arrows: bool = True
    background_color: Property[
        np.ndarray, None, transform_single_color
    ] = np.array([1, 1, 1, 1])
