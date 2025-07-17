import numpy as np
from vispy.scene.node import Node
from vispy.scene.visuals import Polygon, Text
from vispy.util.svg import Document
from vispy.visuals.transforms import STTransform

from napari import __version__
from napari.resources import get_icon_path


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
        self.logo_coords[:, 1] -= 150  # magic number shifting up logo
        super().__init__()

        self.logo = Polygon(
            self.logo_coords, border_method='agg', border_width=2, parent=self
        )
        self.version = Text(
            text='napari',
            pos=[0, 0],
            anchor_x='center',
            anchor_y='bottom',
            parent=self,
        )
        self.shortcuts = Text(
            text='stuff',
            pos=[-240, 50],
            anchor_x='left',
            anchor_y='bottom',
            parent=self,
        )

        self.tips = Text(
            text='stuff',
            pos=[-240, 50],
            anchor_x='left',
            anchor_y='bottom',
            parent=self,
        )

        self.transform = STTransform()

    def set_color(self, color):
        self.logo.color = color
        self.logo.border_color = color
        self.version.color = color
        self.shortcuts.color = color

    def set_text(self):
        self.version.text = f'napari {__version__}'
        self.shortcuts.text = (
            'Some shortcuts\ngo here\nvery cool stuff: do thing'
        )

    def set_scale_and_position(self, x, y):
        self.transform.translate = (x / 2, y / 2, 0, 0)
        scale = min(x, y) * 0.002  # magic number
        self.transform.scale = (scale, scale, 0, 0)
        for text in (self.version, self.shortcuts):
            text.font_size = max(scale * 10, 10)

    def set_gl_state(self, *args, **kwargs):
        for node in self.children:
            node.set_gl_state(*args, **kwargs)
