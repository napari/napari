"""
File with things that are useful for testing, but not to be fixtures
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from napari.utils._proxies import ReadOnlyWrapper


@dataclass
class MouseEvent:
    """Create a subclass for simulating vispy mouse events."""

    type: str
    is_dragging: bool = False
    modifiers: list[str] = field(default_factory=list)
    position: Union[tuple[int, int], tuple[int, int, int]] = (
        0,
        0,
    )  # world coords
    pos: np.ndarray = field(
        default_factory=lambda: np.zeros(2)
    )  # canvas coords
    view_direction: Optional[list[float]] = None
    up_direction: Optional[list[float]] = None
    dims_displayed: list[int] = field(default_factory=lambda: [0, 1])
    delta: Optional[tuple[float, float]] = None
    native: Optional[bool] = None


def read_only_mouse_event(*args, **kwargs):
    return ReadOnlyWrapper(
        MouseEvent(*args, **kwargs), exceptions=('handled',)
    )
