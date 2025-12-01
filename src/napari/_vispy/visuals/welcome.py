from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING, Any

import numpy as np
from vispy.scene.node import Node
from vispy.scene.visuals import Polygon
from vispy.util.svg import Document
from vispy.visuals.transforms import STTransform

from napari._app_model import get_app_model
from napari._vispy.visuals.text import Text
from napari.resources import get_icon_path
from napari.settings import get_settings
from napari.utils.action_manager import action_manager
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
        self.header = Text(
            text='',
            pos=[0, 0],
            anchor_x='center',
            anchor_y='bottom',
            method='gpu',
            parent=self,
        )
        self.shortcut_keybindings = Text(
            text='',
            line_height=1.3,
            pos=[-50, 70],
            anchor_x='right',
            anchor_y='bottom',
            method='gpu',
            parent=self,
        )
        self.shortcut_descriptions = Text(
            text='',
            line_height=1.3,
            pos=[-30, 70],
            anchor_x='left',
            anchor_y='bottom',
            method='gpu',
            parent=self,
        )
        self.tip = Text(
            text='',
            line_height=1.3,
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
        self.header.color = color
        self.shortcut_keybindings.color = color
        self.shortcut_descriptions.color = color
        self.tip.color = color

    def set_version(self, version) -> None:
        self.header.text = (
            f'napari {version}\n\n'
            'Drag file(s) here to open, or use the shortcuts below:'
        )

    def set_shortcuts(self, commands) -> None:
        shortcuts = {}
        app = get_app_model()
        all_shortcuts = get_settings().shortcuts.shortcuts
        for command_id in commands:
            keybinding = app.keybindings.get_keybinding(command_id)
            if keybinding is not None:
                shortcut = Shortcut(keybinding.keybinding).platform
                command = app.commands[command_id].title
            else:
                # might be an action_manager action
                keybinding = all_shortcuts.get(command_id, [None])[0]
                if keybinding is not None:
                    shortcut = Shortcut(keybinding).platform
                    command = action_manager._actions[command_id].description
                else:
                    continue

            shortcuts[shortcut] = command

        self.shortcut_keybindings.text = '\n'.join(shortcuts.keys())
        self.shortcut_descriptions.text = '\n'.join(shortcuts.values())

    def set_tip(self, tip) -> None:
        # TODO: this should use template strings in the future
        for match in re.finditer(r'{(.*?)}', tip):
            command_id = match.group(1)
            app = get_app_model()
            keybinding = app.keybindings.get_keybinding(command_id)
            # this can be none at launch (not yet initialized), will be updated after
            if keybinding is not None:
                shortcut = Shortcut(keybinding.keybinding)
                tip = re.sub(match.group(), str(shortcut), tip)

        # wrap tip so it's not clipped
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
            text.font_size = max(scale * 10, 10)

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        for node in self.children:
            node.set_gl_state(*args, **kwargs)
