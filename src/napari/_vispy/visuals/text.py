from typing import Any

from vispy.scene.visuals import Text as BaseText

from napari._vispy.utils.text import (
    get_text_width_height,
    register_napari_fonts,
)


class Text(BaseText):
    def __init__(
        self, *args: Any, face: str = 'AlataPlus', **kwargs: Any
    ) -> None:
        register_napari_fonts()
        super().__init__(*args, face=face, **kwargs)

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
