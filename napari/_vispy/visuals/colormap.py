import numpy as np
from vispy.scene import Image, Line, Node, STTransform, Text


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
        self.img = Image(parent=self)
        self.img.transform = STTransform()
        self.box = Line(
            pos=self._box_data, connect='strip', method='gl', parent=self
        )
        self.box.transform = STTransform()
        self.ticks = Line(connect='segments', method='gl', parent=self)
        self.ticks.transform = STTransform()
        self.tick_values = []

        self._texture_size = 250
        self.set_data_and_clim()

    def set_data_and_clim(self, clim=(0, 1), dtype=np.float32):
        self.img.set_data(
            np.linspace(
                clim[1], clim[0], self._texture_size, dtype=dtype
            ).reshape(-1, 1)
        )
        self.img.clim = clim

    def set_cmap(self, cmap):
        self.img.cmap = cmap

    def set_gamma(self, gamma):
        self.img.gamma = gamma

    def set_gl_state(self, *args, **kwargs):
        self.img.set_gl_state(*args, **kwargs)

    def set_size(self, size):
        self.img.transform.scale = size[0], size[1] / self._texture_size
        self.box.transform.scale = size

    def set_ticks(self, show, n, tick_length, size, font_size, clim, color):
        self.box.set_data(color=color)
        self.ticks.visible = show
        for text in self.tick_values:
            text.parent = None

        if not show:
            return 0

        ticks_pos = np.linspace(0, 1, n)
        ticks_x = np.tile((size[0], size[0] + tick_length), n)
        ticks_y = np.repeat(ticks_pos, 2) * size[1]
        ticks_coords = np.stack([ticks_x, ticks_y]).T
        self.ticks.set_data(pos=ticks_coords, color=color, width=1)

        ticks_vals = np.interp(1 - ticks_pos, (0, 1), clim)
        max_val_width = 0
        for val, pos in zip(ticks_vals, ticks_coords[1::2], strict=True):
            val_str = f'{val:.3}'
            max_val_width = max(max_val_width, len(val_str))
            # using anchor_y=center doesn't work well for some reason
            # so instead we use top and shift y manually
            text = Text(
                text=val_str,
                pos=pos + (3, font_size / 2),
                color=color,
                anchor_x='left',
                anchor_y='top',
                font_size=font_size,
                parent=self,
            )
            self.tick_values.append(text)

        return max_val_width * font_size
