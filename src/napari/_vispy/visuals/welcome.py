from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from vispy.scene.node import Node
from vispy.scene.visuals import Polygon, Text
from vispy.util.svg import Document
from vispy.visuals.transforms import STTransform

from napari import __version__
from napari.resources import get_icon_path

if TYPE_CHECKING:
    from napari.utils.color import ColorValue


class Welcome(Node):
    def __init__(self) -> None:
        self.logo_coords = (
            Document(get_icon_path('logo_silhouette')).paths[0].vertices[1][0]
        )
        self.logo_coords = self.logo_coords[
            :, :2
        ]  # drop z: causes issues with polygon agg mode
        # center vertically and move up
        self.logo_coords -= (
            np.max(self.logo_coords, axis=0) + np.min(self.logo_coords, axis=0)
        ) / 2
        self.logo_coords[:, 1] -= 130  # magic number shifting up logo
        super().__init__()

        self.logo = Polygon(
            self.logo_coords, border_method='agg', border_width=2, parent=self
        )
        self.version = Text(
            text='',
            pos=[0, 0],
            anchor_x='center',
            anchor_y='bottom',
            method='gpu',
            parent=self,
        )
        self.shortcuts = Text(
            text='',
            pos=[-240, 50],
            anchor_x='left',
            anchor_y='bottom',
            method='gpu',
            parent=self,
        )
        self.tips = Text(
            text='',
            pos=[0, 180],
            anchor_x='center',
            anchor_y='bottom',
            method='gpu',
            parent=self,
        )

        self.transform = STTransform()

    def set_color(self, color: ColorValue) -> None:
        self.logo.color = color
        self.logo.border_color = color
        self.version.color = color
        self.shortcuts.color = color
        self.tips.color = color

    def set_text(self) -> None:
        self.version.text = f'napari {__version__}'
        self.shortcuts.text = (
            'Drag image(s) here to open or use the shortcuts below:\n\n'
            # TODO: these need to be system specific
            'Ctrl+N: New Image from Clipboard\n'
            'Ctrl+O: Open image(s)\n'
            'Ctrl+Shift+P: Show Command Palette\n'
        )
        self.tips.text = 'This is a tip'

    def set_scale_and_position(self, x: float, y: float) -> None:
        self.transform.translate = (x / 2, y / 2, 0, 0)
        scale = min(x, y) * 0.002  # magic number
        self.transform.scale = (scale, scale, 0, 0)

        # update the dpi scale factor to account for screen dpi
        # because vispy scales pixel height of text by screen dpi
        if self.transforms.dpi:
            # use 96 as the napari reference dpi for historical reasons
            dpi_scale_factor = 96 / self.transforms.dpi
        else:
            dpi_scale_factor = 1

        for text in (self.version, self.shortcuts, self.tips):
            text.font_size = max(scale * 10 * dpi_scale_factor, 10)

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        for node in self.children:
            node.set_gl_state(*args, **kwargs)
