from vispy.scene.visuals import Text as BaseText

from napari._vispy.utils.text import get_text_width_height


class Text(BaseText):
    def get_width_height(self):
        # adjust for dpi: 72 is the "base dpi" around which font sizes are defined
        width, height = get_text_width_height(self)
        return width * self.dpi_ratio, height * self.dpi_ratio

    @property
    def font_size(self):
        return self._font_size

    @font_size.setter
    def font_size(self, size):
        # adjust for dpi: 72 is the "base dpi" around which font sizes are defined
        adjusted_font_size = size / self.dpi_ratio
        self._font_size = max(0.0, adjusted_font_size)
        self.update()

    @property
    def dpi_ratio(self):
        return (self.transforms.dpi or 72) / 72
