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
        return get_text_width_height(self)

    @property
    def font_size(self) -> float:
        return self._font_size

    @font_size.setter
    def font_size(self, size: float) -> None:
        adjusted_font_size = size / self.dpi_ratio
        self._font_size = max(0.0, adjusted_font_size)
        self.update()

    @property
    def dpi_ratio(self) -> float:
        # adjust for dpi: 72 is the "base dpi" around which font sizes are defined
        return (self.transforms.dpi or 72) / 72
