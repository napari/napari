from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vispy.scene import Axis, Image, Line, Node, STTransform

from napari._vispy.visuals.text import Text

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt
    from vispy.color import Colormap as VispyColormap

    from napari.utils.color import ColorValue


class Colormap(Node):
    _box_data = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 0],
        ]
    )

    def __init__(self) -> None:
        super().__init__()
        self.ticks = Axis(
            pos=self._box_data[2:4],
            tick_direction=(1, 0),
            tick_width=2,
            tick_label_margin=4,
            axis_width=0,
            parent=self,
        )
        # override to use our class which works better with hidpi
        self.ticks.remove_subvisual(self.ticks._text)
        self.ticks._text = Text(
            font_size=self.ticks.tick_font_size, color=self.ticks.text_color
        )
        self.ticks.add_subvisual(self.ticks._text)

        self.ticks.transform = STTransform()
        self.img = Image(parent=self)
        self.img.transform = STTransform()
        self.box = Line(
            pos=self._box_data,
            connect='strip',
            method='gl',
            width=1,
            parent=self,
        )
        self.box.transform = STTransform()

        self._texture_size = 250
        self._text_vertices_size = (0, 0)
        self.set_data_and_clim()

    def set_data_and_clim(
        self,
        clim: tuple[float, float] = (0, 1),
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        self.img.set_data(
            np.linspace(
                clim[1], clim[0], self._texture_size, dtype=dtype
            ).reshape(-1, 1)
        )
        self.img.clim = clim

    def set_cmap(self, cmap: VispyColormap) -> None:
        self.img.cmap = cmap

    def set_gamma(self, gamma: float) -> None:
        self.img.gamma = gamma

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        self.img.set_gl_state(*args, **kwargs)

    def set_size(self, size: tuple[float, float]) -> None:
        self.img.transform.scale = size[0], size[1] / self._texture_size
        self.box.transform.scale = size
        self.ticks.transform.scale = size

    def set_ticks_and_get_text_size(
        self,
        tick_length: float,
        font_size: int,
        clim: tuple[float, float],
        color: ColorValue,
    ) -> tuple[float, float]:
        self.box.set_data(color=color)
        self.ticks.axis_color = color
        self.ticks.text_color = color
        self.ticks.tick_color = color
        self.ticks.tick_font_size = font_size
        self.ticks.major_tick_length = tick_length
        self.ticks.minor_tick_length = tick_length / 2
        self.ticks.domain = clim

        text = self.ticks._text
        self.ticks._update_subvisuals()  # triggers computing of the tick labels

        width, height = text.get_width_height()
        return width + self.ticks.tick_label_margin, height
