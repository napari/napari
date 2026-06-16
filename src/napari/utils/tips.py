"""
This module contains a list of tips and tricks that will be displayed to napari users
in a variety of places (e.g: the welcome screen shows a random one on each startup.).
Tips should added as an extra string inside the `NAPARI_TIPS` tuple. They may contain
format interpolators (in the form `{something}`) which will be replaced as appropriate
by the `format_tip()` function. Allowed interpolators are:

- a command_id (e.g. napari.window.view.toggle_viewer_axes) which will be replaced
  by the relative platform-specific shortcut
- a keybinding (e.g. Ctrl+Y) for when the above is not possible because the action
  is not implemented as an app_model command (e.g: increase/decrease brush size)

Community contributions are very welcome!
"""

# TODO: add to options: a URL (e.g: https://napari.org) which will become a hyperlink where possible

import re
import warnings
from typing import Final

from napari.utils.interactions import Shortcut

NAPARI_TIPS: Final[tuple[str, ...]] = (
    'You can take a screenshot of the canvas and copy it to your clipboard by pressing {napari.window.file.copy_canvas_screenshot}.',
    'You can change most shortcuts from the File → Preferences → Shortcuts menu.',
    'You can right click many components of the graphical interface to access advanced controls.',
    'If you select multiple layers in the layer list, then right click and select "Link Layers", their parameters will be synced.',
    'You can press {Ctrl} and scroll the mouse wheel to move the dimension sliders.',
    'To zoom in on a specific area, hold {Alt} and draw a rectangle around it.',
    'Hold {napari:hold_for_pan_zoom} to pan/zoom in any mode (e.g. while painting).',
    'While painting labels, hold {Alt} and move the cursor left/right to quickly decrease/increase the brush size.',
    'If you have questions, you can reach out on our community chat at napari.zulipchat.com!',
    'The community at forum.image.sc is full of imaging experts sharing knowledge and tools for napari and much, much more!',
)


def _get_command_shortcut_and_description(
    command_id: str,
) -> tuple[str | None, str | None]:
    """Get the command shortcut and description from a command id."""
    from napari._app_model import get_app_model
    from napari.settings import get_settings
    from napari.utils.action_manager import action_manager

    app = get_app_model()

    if app_model_keybinding := app.keybindings.get_keybinding(command_id):
        return (
            Shortcut(app_model_keybinding.keybinding).platform,
            app.commands[command_id].title,
        )

    # might be an action_manager action
    settings_shortcuts = get_settings().shortcuts.shortcuts
    if action_manager_keybinding := settings_shortcuts.get(command_id, [None])[
        0
    ]:
        return (
            Shortcut(action_manager_keybinding).platform,
            action_manager._actions[command_id].description,
        )

    return None, None


def format_tip(tip: str) -> str:
    """Format a tip with appropriate shortcuts and command keybindings.

    Input tip should use template string formatting.
    Expands to platform-specific glyphs as appropriate.
    Accepts direct shortcuts such as `{Alt+X}` and command ids
    such as `napari.window.file.copy_canvas_screenshot`.

    Note: for some actions, the napari viewer needs to be initialized once
    in order for them to be registered.
    """
    # TODO: this should use template strings in the future
    for match in re.finditer(r'{(.*?)}', tip):
        command_id = match.group(1)
        shortcut, _ = _get_command_shortcut_and_description(command_id)
        # this can be none at launch (not yet initialized), will be updated after
        if shortcut is None:
            # maybe it was just a direct keybinding given
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                shortcut = Shortcut(command_id).platform
        if shortcut:
            tip = tip.replace(match.group(), str(shortcut))
    return tip
