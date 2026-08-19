import warnings

import numpy as np

from napari.utils.events import EventedModel


class Cursor(EventedModel):
    """Cursor object with position and properties of the cursor.

    Attributes
    ----------
    position : tuple of float
        Position of the cursor in world coordinates. If the cursor is outside of,
        the canvas, then the last known position is stored instead.
    viewbox : tuple[int, int] or None
        Position of the cursor in the grid.
    _view_direction : Optional[np.ndarray]
        The vector describing the direction of the camera in the scene
        in world coordinates.
        This is None when viewing in 2D.
    """

    # fields
    position: tuple[float, ...] = (1.0, 1.0)
    canvas_position: tuple[int, int] = (1, 1)
    viewbox: tuple[int, int] | None = None
    _view_direction: np.ndarray | None = None

    @property
    def style(self) -> None:
        warnings.warn(
            'cursor.style is deprecated since 0.10.0 and will be removed in a future '
            'version. The cursor style is now determined by the active layer.'
        )
        return

    @style.setter
    def style(self, style: str) -> None:
        warnings.warn(
            'cursor.style is deprecated since 0.10.0 and will be removed in a future '
            'version. The cursor style is now determined by the active layer.'
        )
        self._style = style

    @property
    def size(self) -> float:
        warnings.warn(
            'cursor.size is deprecated since 0.10.0 and will be removed in a future '
            'version. The cursor size is now determined by the active layer.'
        )
        return self._size

    @size.setter
    def size(self, size: float) -> None:
        warnings.warn(
            'cursor.size is deprecated since 0.10.0 and will be removed in a future '
            'version. The cursor size is now determined by the active layer.'
        )
        self._size = size

    @property
    def scaled(self) -> bool:
        warnings.warn(
            'cursor.scaled is deprecated since 0.10.0 and will be removed in a future '
            'version. The cursor scaling is now determined by the active layer.'
        )
        return self._scaled

    @scaled.setter
    def scaled(self, scaled: bool) -> None:
        warnings.warn(
            'cursor.scaled is deprecated since 0.10.0 and will be removed in a future '
            'version. The cursor scaling is now determined by the active layer.'
        )
        self._scaled = scaled
