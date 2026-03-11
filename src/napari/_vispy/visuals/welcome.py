from __future__ import annotations

import logging
import re
import textwrap
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from qtpy.QtGui import QFont, QGuiApplication
from vispy.scene.node import Node
from vispy.scene.visuals import Image, Polygon
from vispy.util.svg import Document
from vispy.visuals.transforms import STTransform

from napari._app_model import get_app_model
from napari.resources import get_icon_path
from napari.settings import get_settings
from napari.utils.action_manager import action_manager
from napari.utils.interactions import Shortcut

if TYPE_CHECKING:
    from napari.utils.color import ColorValue

vispy_logger = logging.getLogger('vispy')


class Welcome(Node):
    _BASE_FONT_SIZE = 14
    _LINE_HEIGHT = 1.4
    _TEXT_PADDING = 2
    # this should probably match HiDPI/UI scaling
    _TEXT_RASTER_SCALE = 2

    def __init__(self) -> None:
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
        self.logo.transform = STTransform()
        self.text_image = Image(
            np.zeros((1, 1, 4), dtype=np.uint8), parent=self
        )
        self.text_image.transform = STTransform()
        self.text_image.interpolation = 'linear'

        self.transform = STTransform()
        self._scale = 1.0
        self._text_origin = np.zeros(2, dtype=float)
        self._text_color = (255, 255, 255, 255)
        self._text_raster_cache_key: tuple[Any, ...] | None = None

        self._header = ''
        self._shortcut_keys = ''
        self._shortcut_descriptions = ''
        self._tip = ''

    def set_color(self, color: ColorValue) -> None:
        self.logo.color = color
        self.logo.border_color = color
        rgba = np.clip(np.asarray(color), 0, 1)
        if len(rgba) == 3:
            rgba = np.append(rgba, 1)
        self._text_color = tuple((rgba[:4] * 255).astype(int))
        self._update_text_texture()

    def set_version(self, version: str) -> None:
        self._header = (
            f'napari {version}\n\n'
            'Drag file(s) here to open, or use the shortcuts below:'
        )
        self._update_text_texture()

    def set_shortcuts(self, commands: tuple[str, ...]) -> None:
        shortcuts = {}
        for command_id in commands:
            shortcut, command = self._command_shortcut_and_description(
                command_id
            )
            if shortcut is not None and command is not None:
                shortcuts[shortcut] = command

        # TODO: use template strings in the future
        self._shortcut_keys = '\n'.join(shortcuts.keys())
        self._shortcut_descriptions = '\n'.join(shortcuts.values())
        self._update_text_texture()

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

        # wrap tip so it's not clipped; width reduced proportionally with font size
        self._tip = 'Did you know?\n' + '\n'.join(
            textwrap.wrap(tip, width=60, break_on_hyphens=False)
        )
        self._update_text_texture()

    def _font_compensation(self) -> float:
        """Return a size multiplier to keep text crisp at small canvas scales.

        This counteracts the actual scaling/transform, preventing text from
        getting too small, aiming to keep the text size near the default
        font size for legibility.
        Note: Capped at 8x to avoid absurdly large textures when the canvas is
        extremely small (e.g. a minimised or unit-test window).
        """
        if self._scale <= 0:
            return 1
        if self._scale < 1:
            return min(8.0, round(1 / self._scale, 2))
        return 1

    def _text_blocks(
        self,
    ) -> tuple[
        tuple[str, float, float, Literal['left', 'center', 'right']], ...
    ]:
        """Return text content with anchor points and alignment in local coords.

        Coordinate system
        -----------------
        The canvas is centred at (0, 0) with y increasing *downward* (screen
        coords).  Each ``(text, anchor_x, anchor_y, align)`` tuple places the
        *bottom* edge of a text block at ``(anchor_x, anchor_y)``.  Text
        extends *upward* (to more-negative y) from that anchor.  To move a
        block lower on screen, increase anchor_y; to raise it, decrease it.
        """
        return (
            (self._header, 0, 40, 'center'),
            (self._shortcut_keys, -95, 120, 'right'),
            (self._shortcut_descriptions, -70, 120, 'left'),
            (self._tip, 0, 220, 'center'),
        )

    def _render_text_texture(self) -> tuple[np.ndarray, tuple[float, float]]:
        """Render current welcome text into a Qt-backed RGBA texture and origin."""
        from napari._qt.utils import rasterize_text_blocks_to_array

        raster_scale = self._TEXT_RASTER_SCALE
        font = QFont(QGuiApplication.font())
        # Safely try to set Antialiasing, if not set
        prefer_antialias = getattr(QFont, 'PreferAntialias', None)
        if prefer_antialias is None:
            style_strategy_enum = getattr(QFont, 'StyleStrategy', None)
            if style_strategy_enum is not None:
                prefer_antialias = getattr(
                    style_strategy_enum, 'PreferAntialias', None
                )
        if prefer_antialias is not None:
            font.setStyleStrategy(font.styleStrategy() | prefer_antialias)
        font.setPixelSize(
            max(
                1,
                round(
                    self._BASE_FONT_SIZE
                    * self._font_compensation()
                    * raster_scale
                ),
            )
        )
        return rasterize_text_blocks_to_array(
            self._text_blocks(),
            font=font,
            line_height=self._LINE_HEIGHT,
            color=self._text_color,
            raster_scale=raster_scale,
            padding=self._TEXT_PADDING,
        )

    def _update_text_texture(self) -> None:
        """Re-rasterize text only when cached content/style inputs changed."""
        cache_key = (
            self._header,
            self._shortcut_keys,
            self._shortcut_descriptions,
            self._tip,
            self._text_color,
            self._font_compensation(),
        )
        if cache_key == self._text_raster_cache_key:
            self._update_text_transform()
            return

        text_data, origin = self._render_text_texture()
        self.text_image.set_data(text_data)
        self._text_origin = np.asarray(origin)
        self._text_raster_cache_key = cache_key
        self._update_text_transform()

    def _update_text_transform(self) -> None:
        """Apply scene transform for the raster texture using local text origin."""
        scale = self._scale
        raster_scale = self._TEXT_RASTER_SCALE
        self.text_image.transform.scale = (
            scale / raster_scale,
            scale / raster_scale,
            1,
            1,
        )
        self.text_image.transform.translate = (
            self._text_origin[0] * scale,
            self._text_origin[1] * scale,
            0,
            0,
        )

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
        self.transform.translate = (x / 2, y / 2, 0, 0)
        scale = min(x, y) * 0.002  # magic number
        self._scale = scale
        self.logo.transform.scale = (scale, scale, 1, 1)
        self._update_text_texture()

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        for node in self.children:
            node.set_gl_state(*args, **kwargs)
