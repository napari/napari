from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
from vispy.scene.node import Node
from vispy.scene.visuals import Polygon
from vispy.util.svg import Document
from vispy.visuals.transforms import STTransform

from napari._app_model import get_app_model
from napari._vispy.visuals.text import Text
from napari.resources import get_icon_path
from napari.utils.interactions import Shortcut

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
        self.tip = Text(
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
        self.tip.color = color

    def set_version(self, version) -> None:
        self.version.text = f'napari {version}'

    def set_shortcuts(self, commands) -> None:
        app = get_app_model()
        shortcuts = {}
        for command_id in commands:
            keybinding = app.keybindings.get_keybinding(command_id)
            # this can be none at launch (not yet initialized), will be updated after
            if keybinding is not None:
                shortcut = Shortcut(keybinding.keybinding)
                command = app.commands[command_id].title
                shortcuts[shortcut] = command

        self.shortcuts.text = (
            'Drag file(s) here to open, or use the shortcuts below:\n\n'
            + '\n'.join(
                f'{shortcut}: {command}'
                for shortcut, command in shortcuts.items()
            )
        )

    def set_tip(self, tip) -> None:
        # this should use template strings in the future
        for match in re.finditer(r'{(.*?)}', tip):
            command_id = match.group(1)
            app = get_app_model()
            keybinding = app.keybindings.get_keybinding(command_id)
            # this can be none at launch (not yet initialized), will be updated after
            if keybinding is not None:
                shortcut = Shortcut(keybinding.keybinding)
                tip = re.sub(match.group(), str(shortcut), tip)
        self.tip.text = 'Did you know?\n' + tip

    def set_scale_and_position(self, x: float, y: float) -> None:
        self.transform.translate = (x / 2, y / 2, 0, 0)
        scale = min(x, y) * 0.002  # magic number
        self.transform.scale = (scale, scale, 0, 0)

        for text in (self.version, self.shortcuts, self.tip):
            text.font_size = max(scale * 8, 10)

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        for node in self.children:
            node.set_gl_state(*args, **kwargs)
