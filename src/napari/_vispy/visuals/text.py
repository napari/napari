from vispy.scene.visuals import Text as BaseText

from napari._vispy.utils.text import get_text_width_height


class Text(BaseText):
    def get_width_height(self) -> tuple[float, float]:
        width, height = get_text_width_height(self)
        # width is not quite right for some reason... magic number here we go
        return width * self.dpi_ratio * 1.2, height * self.dpi_ratio

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
        # TODO: but for some reason 96 seems to give the correct ratio for me?
        return (self.transforms.dpi or 96) / 96
