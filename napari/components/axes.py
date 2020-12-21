from dataclasses import field

import numpy as np

# from pydantic import validator
from pydantic.dataclasses import dataclass

# from ..utils.colormaps.standardize_color import transform_single_color
from ..utils.events.event_utils import PydanticConfig, evented


class _ArrayMeta(type):
    def __getitem__(self, t):
        return type('Array', (Array,), {'__dtype__': t})


class Array(np.ndarray, metaclass=_ArrayMeta):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        dtype = getattr(cls, '__dtype__', None)
        if isinstance(dtype, tuple):
            dtype, shape = dtype
        else:
            shape = tuple()

        result = np.array(val, dtype=dtype, copy=False, ndmin=len(shape))
        assert not shape or len(shape) == len(
            result.shape
        )  # ndmin guarantees this

        if any(
            (shape[i] != -1 and shape[i] != result.shape[i])
            for i in range(len(shape))
        ):
            result = result.reshape(shape)
        return result


def make_default_color_array():
    return np.array([1, 1, 1, 1])


@evented
@dataclass(config=PydanticConfig)
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
    background_color: Array[float, (-1, 4)] = field(
        default_factory=make_default_color_array
    )

    # @validator('background_color', pre=True)
    # def _ensure_color(cls, v):
    #     return transform_single_color(v)
