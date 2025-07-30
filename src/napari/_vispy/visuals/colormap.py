from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vispy.scene import Axis, Image, Line, Node, STTransform

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
            tick_width=1,
            axis_width=0,
            parent=self,
        )
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

    def set_ticks_and_get_text_width(
        self,
        size: tuple[float, float],
        tick_length: float,
        font_size: int,
        clim: tuple[float, float],
        color: ColorValue,
    ) -> float:
        self.box.set_data(color=color)
        self.ticks.axis_color = color
        self.ticks.text_color = color
        self.ticks.tick_color = color
        self.ticks.tick_font_size = font_size
        self.ticks.major_tick_length = tick_length
        self.ticks.minor_tick_length = tick_length / 2
        self.ticks.domain = clim

        text = self.ticks._text
        if text.transforms.dpi:
            # use 96 as the napari reference dpi for historical reasons
            dpi_scale_factor = 96 / text.transforms.dpi
        else:
            dpi_scale_factor = 1

        font_size *= dpi_scale_factor

        vert_buffer = text._vertices_data
        if vert_buffer is not None:
            pos = vert_buffer['a_position']
            tl = pos.min(axis=0)
            br = pos.max(axis=0)
            self._text_vertices_size = (br[0] - tl[0]), (br[1] - tl[1])

        text_width = self._text_vertices_size[0] * font_size * 1.3  # magic?
        # fixed multiplier for height to avoid fluttering when zooming
        text_height = font_size * 1.5

        return text_width, text_height
