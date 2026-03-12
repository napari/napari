from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING, Any

import numpy as np
from vispy.scene.node import Node
from vispy.scene.visuals import Polygon
from vispy.util.svg import Document
from vispy.visuals.transforms import STTransform

from napari._app_model.utils import get_command_shortcut_and_description
from napari._vispy.visuals.text import Text
from napari.resources import get_icon_path
from napari.utils.tips import format_tip

if TYPE_CHECKING:
    from vispy.visuals.text.text import FontManager

    from napari.utils.color import ColorValue

vispy_logger = logging.getLogger('vispy')


class Welcome(Node):
    def __init__(self, font_manager: FontManager, face: str) -> None:
        old_level = vispy_logger.level
        vispy_logger.setLevel(logging.ERROR)
        self.logo_coords = (
            Document(get_icon_path('logo_silhouette')).paths[0].vertices[0][0]
        )
        vispy_logger.setLevel(old_level)
        self.logo_coords = self.logo_coords[
            :, :2
        ]  # drop z: causes issues with polygon agg mode
        # center
        self.logo_coords -= (
            np.max(self.logo_coords, axis=0) + np.min(self.logo_coords, axis=0)
        ) / 2
        # make it smaller
        self.logo_coords /= 4
        # move it up
        self.logo_coords[:, 1] -= 130  # magic number shifting up logo
        super().__init__()

        self.logo = Polygon(
            self.logo_coords, border_method='agg', border_width=2, parent=self
        )
        self.header = Text(
            text='',
            pos=[0, 0],
            anchor_x='center',
            anchor_y='bottom',
            parent=self,
            font_manager=font_manager,
            face=face,
        )
        self.shortcut_keybindings = Text(
            text='',
            line_height=1.15,
            pos=[-80, 60],
            anchor_x='right',
            anchor_y='bottom',
            parent=self,
            font_manager=font_manager,
            face=face,
        )
        self.shortcut_descriptions = Text(
            text='',
            line_height=1.15,
            pos=[-60, 60],
            anchor_x='left',
            anchor_y='bottom',
            parent=self,
            font_manager=font_manager,
            face=face,
        )
        self.tip = Text(
            text='',
            line_height=1.15,
            pos=[0, 160],
            anchor_x='center',
            anchor_y='bottom',
            parent=self,
            font_manager=font_manager,
            face=face,
        )

        self.transform = STTransform()

    def set_color(self, color: ColorValue) -> None:
        self.logo.color = color
        self.logo.border_color = color
        self.header.color = color
        self.shortcut_keybindings.color = color
        self.shortcut_descriptions.color = color
        self.tip.color = color

    def set_version(self, version: str) -> None:
        self.header.text = (
            f'napari {version}\n\n'
            'Drag file(s) here to open, or use the shortcuts below:'
        )

    def set_shortcuts(self, commands: tuple[str, ...]) -> None:
        shortcuts = {}
        for command_id in commands:
            shortcut, command = get_command_shortcut_and_description(
                command_id
            )
            if shortcut is not None and command is not None:
                shortcuts[shortcut] = command

        self.shortcut_keybindings.text = '\n'.join(shortcuts.keys())
        self.shortcut_descriptions.text = '\n'.join(shortcuts.values())

    def set_tip(self, tip: str) -> None:
        # wrap tip so it's not clipped
        tip = format_tip(tip)
        self.tip.text = 'Did you know?\n' + '\n'.join(
            textwrap.wrap(tip, break_on_hyphens=False)
        )

    def set_scale_and_position(self, x: float, y: float) -> None:
        self.transform.translate = (x / 2, y / 2, 0, 0)
        scale = min(x, y) * 0.002  # magic number
        self.transform.scale = (scale, scale, 0, 0)

        for text in (
            self.header,
            self.shortcut_keybindings,
            self.shortcut_descriptions,
            self.tip,
        ):
            text.font_size = max(scale * 8, 10)

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        for node in self.children:
            node.set_gl_state(*args, **kwargs)
