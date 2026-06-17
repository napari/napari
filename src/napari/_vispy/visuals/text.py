from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vispy.scene.visuals import Text as BaseText

from napari._vispy.utils.text import (
    get_text_width_height,
)

# Global Qt-based font manager instance shared across all Text visuals

_FONT_FAMILY = 'OpenSans'

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo


class Text(BaseText):
    def __init__(
        self, *args: Any, font_info: FontInfo | None = None, **kwargs: Any
    ) -> None:
        # If using Qt fonts, pass the Qt font manager to the base class
        if font_info is not None:
            kwargs['font_manager'] = font_info.font_manager
            kwargs['face'] = font_info.face

        super().__init__(*args, **kwargs)

    def get_width_height(self) -> tuple[float, float]:
        width, height = get_text_width_height(self)
        return width * self._vispy_dpi_ratio, height * self._vispy_dpi_ratio

    @property
    def font_size(self) -> float:
        return self._font_size * self.dpi_ratio

    @font_size.setter
    def font_size(self, size: float) -> None:
        adjusted_font_size = size / self.dpi_ratio
        self._font_size = max(0.0, adjusted_font_size)
        self.update()

    @property
    def _vispy_dpi_ratio(self) -> float:
        # while 72 is considered the base dpi in vispy, most software assumes
        # 96 as reference dpi, resulting in mismatch between gui and canvas
        # when font sizes are the same. We need to account for this when
        # we calculate things externally from vispy and
        return 96 / 72

    @property
    def dpi_ratio(self) -> float:
        dpi = self.transforms.dpi or 96
        return dpi / 96
