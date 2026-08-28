from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vispy.scene.visuals import Line

from napari._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin

if TYPE_CHECKING:
    import numpy as np


class Vectors(ClippingPlanesMixin, Line):
    """Fallback line-based vectors visual."""

    def __init__(self, **kwargs: Any):
        super().__init__(connect='segments', **kwargs)

    def set_data(
        self,
        vertices: np.ndarray,
        colors: np.ndarray,
        vector_style: str,
    ) -> None:
        super().set_data(pos=vertices, color=colors.repeat(2, axis=0))

    @property
    def width(self) -> float:
        return Line.width.fget(self)

    @width.setter
    def width(self, value: float) -> None:
        pass
        # TODO: this width is in screen pixels; we shoould make this
        #       an option for normal vectors too, and then at least this
        #       would work!
        # super().set_data(width=value)
