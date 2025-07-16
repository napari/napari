import numpy as np
from vispy.scene.visuals import Compound, Polygon, Text
from vispy.util.svg import Document
from vispy.visuals.transforms import STTransform

from napari import __version__
from napari.resources import get_icon_path


class Welcome(Compound):
    def __init__(self) -> None:
        self.logo = (
            Document(get_icon_path('logo_silhouette')).paths[0].vertices[1][0]
        )
        self.logo = self.logo[
            :, :2
        ]  # drop z: causes issues with polygon agg mode
        # center vertically and move up
        self.logo -= (
            np.max(self.logo, axis=0) + np.min(self.logo, axis=0)
        ) / 2
        self.logo[:, 1] -= 150
        super().__init__(
            [
                Polygon(self.logo, border_method='agg', border_width=1),
                Text(
                    text='',
                    pos=[0.5, 0.5],
                    anchor_x='center',
                    anchor_y='top',
                    font_size=10,
                ),
            ]
        )

        self.transform = STTransform()

    def set_color(self, color):
        self._subvisuals[0].color = color
        self._subvisuals[0].border_color = color
        self._subvisuals[1].color = color

    def set_text(self):
        self._subvisuals[1].text = f'napari {__version__}'
