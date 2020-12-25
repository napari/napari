from pydantic import Field, validator

from ..utils.colormaps.colormap_utils import make_default_color_array
from ..utils.colormaps.standardize_color import transform_single_color
from ..utils.pydantic import Array, ConfiguredModel, evented_model


@evented_model
class Axes(ConfiguredModel):
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

    # fields
    visible: bool = False
    labels: bool = True
    colored: bool = True
    dashed: bool = False
    arrows: bool = True
    background_color: Array[float, (4,)] = Field(
        default_factory=make_default_color_array
    )

    # validators
    _ensure_single_color = validator(
        'background_color', pre=True, allow_reuse=True
    )(transform_single_color)
