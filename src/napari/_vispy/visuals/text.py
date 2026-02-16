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
        return width * self.dpi_ratio, height * self.dpi_ratio

    @property
    def font_size(self) -> float:
        return self._font_size * self.dpi_ratio / (96 / 72)

    @font_size.setter
    def font_size(self, size: float) -> None:
        # 72 is the "base dpi" for vispy font sizes, but normally 96
        # is considered the base dpi, resulting in mismatch between
        # gui and canvas when font sizes are the same
        adjusted_font_size = size * (96 / 72) / self.dpi_ratio
        self._font_size = max(0.0, adjusted_font_size)
        self.update()

    @property
    def dpi_ratio(self) -> float:
        # 72 is the reference DPI around which vispy font sizes are defined
        return (self.transforms.dpi or 72) / 72
