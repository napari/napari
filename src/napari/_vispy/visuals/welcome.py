from __future__ import annotations

import logging
import re
import textwrap
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from vispy.scene.node import Node
from vispy.scene.visuals import Polygon
from vispy.util.svg import Document
from vispy.visuals.transforms import STTransform

from napari._app_model import get_app_model
from napari._vispy.utils.text import get_text_metrics
from napari._vispy.visuals.text import Text
from napari.resources import get_icon_path
from napari.settings import get_settings
from napari.utils.action_manager import action_manager
from napari.utils.interactions import Shortcut

if TYPE_CHECKING:
    from vispy.visuals.text.text import FontManager

    from napari.utils.color import ColorValue

vispy_logger = logging.getLogger('vispy')


def _load_logo() -> np.ndarray:
    # load logo (disabling logging for some svg reading warnings)
    old_level = vispy_logger.level
    vispy_logger.setLevel(logging.ERROR)
    coords = Document(get_icon_path('logo_silhouette')).paths[0].vertices[0][0]
    vispy_logger.setLevel(old_level)
    # drop z: causes issues with polygon agg mode
    coords = coords[:, :2]
    # center
    coords -= (np.max(coords, axis=0) + np.min(coords, axis=0)) / 2
    return coords


class Welcome(Node):
    def __init__(self, font_manager: FontManager, face: str) -> None:
        self.logo_coords = _load_logo()
        super().__init__()

        # make logo smaller and move it up (magic number)
        self.logo_coords /= 4
        self.logo_coords[:, 1] -= 130

        self.logo = Polygon(
            self.logo_coords, border_method='agg', border_width=2, parent=self
        )
        self.logo.transform = STTransform()

        self.text = Text(
            line_height=1.75,
            anchor_x='center',
            anchor_y='bottom',
            parent=self,
            font_manager=font_manager,
            face=face,
        )
        self.text.transform = STTransform()

        self.font_height = get_text_metrics(self.text).height()

        self.text_components = {
            'version': '',
            'shortcut_keybindings': '',
            'shortcut_descriptions': '',
            'tip': '',
        }
        self._update_text()
        self.text.pos = [[0, 0], [10, 10], [20, 20], [30, 30]]

    def set_color(self, color: ColorValue) -> None:
        self.logo.color = color
        self.logo.border_color = color
        self.text.color = color

    def _update_text(self) -> None:
        self.text.text = list(self.text_components.values())

    def set_version(self, version: str) -> None:
        self.text_components['version'] = (
            f'napari {version}\n'
            'Drag file(s) here to open, or use the shortcuts below:'
        )
        self._update_text()

    def set_shortcuts(self, commands: tuple[str, ...]) -> None:
        shortcuts = {}
        for command_id in commands:
            shortcut, command = self._command_shortcut_and_description(
                command_id
            )
            if shortcut is not None and command is not None:
                shortcuts[shortcut] = command

        # TODO: use template strings in the future
        self.text_components['shortcut_keybindings'] = '\n'.join(
            shortcuts.keys()
        )
        self.text_components['shortcut_descriptions'] = '\n'.join(
            shortcuts.values()
        )
        self._update_text()

    def set_tip(self, tip: str) -> None:
        # TODO: this should use template strings in the future
        for match in re.finditer(r'{(.*?)}', tip):
            command_id = match.group(1)
            shortcut, _ = self._command_shortcut_and_description(command_id)
            # this can be none at launch (not yet initialized), will be updated after
            if shortcut is None:
                # maybe it was just a direct keybinding given
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    shortcut = Shortcut(command_id).platform
            if shortcut:
                tip = re.sub(match.group(), str(shortcut), tip)

        # wrap tip so it's not clipped
        self.text_components['tip'] = 'Did you know?\n' + '\n'.join(
            textwrap.wrap(tip, break_on_hyphens=False)
        )
        self._update_text()

    @staticmethod
    def _command_shortcut_and_description(
        command_id: str,
    ) -> tuple[str | None, str | None]:
        app = get_app_model()
        all_shortcuts = get_settings().shortcuts.shortcuts
        keybinding = app.keybindings.get_keybinding(command_id)

        shortcut = command = None
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
                shortcut = command = None

        return shortcut, command

    def set_scale_and_position(self, x: float, y: float) -> None:
        trans = (x / 2, y / 2, 0, 0)
        # we don't want the logo to be affected by dpi ratio which is included in
        # font_height, so we scale it separately
        logo_scale = min(x, y) * 0.002  # magic number
        self.logo.transform.translate = trans
        self.logo.transform.scale = (logo_scale, logo_scale, 0, 0)

        text_scale = min(x, y) / self.font_height * 0.04  # magic number
        self.text.font_size = text_scale * 8
        self.text.transform.translate = trans
        self.text.transform.scale = (text_scale, text_scale, 0, 0)

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        for node in self.children:
            node.set_gl_state(*args, **kwargs)
