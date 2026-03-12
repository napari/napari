from typing import Any

from vispy.scene.visuals import Text as BaseText

from napari._vispy.utils.qt_font import QtFontManager
from napari._vispy.utils.text import (
    get_text_width_height,
)

# Global Qt-based font manager instance shared across all Text visuals
_qt_font_manager = None

_FONT_FAMILY = 'OpenSans'


def get_qt_font_manager(method: str = 'cpu') -> QtFontManager:
    """Get or create the global Qt font manager instance.

    Parameters
    ----------
    method : str, optional
        Rendering method ('cpu' or 'gpu'). Default is 'cpu'.

    Returns
    -------
    QtFontManager
        The global Qt font manager instance.
    """
    global _qt_font_manager
    if _qt_font_manager is None:
        _qt_font_manager = QtFontManager(method=method)
    return _qt_font_manager


class Text(BaseText):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # If using Qt fonts, pass the Qt font manager to the base class
        if 'face' not in kwargs:
            kwargs['face'] = _FONT_FAMILY
        super().__init__(*args, **kwargs)

    def get_width_height(self) -> tuple[float, float]:
        width, height = get_text_width_height(self)
        return width, height

    @property
    def font_size(self) -> float:
        return self._font_size

    @font_size.setter
    def font_size(self, size: float) -> None:
        self._font_size = max(0.0, size)
        self.update()
