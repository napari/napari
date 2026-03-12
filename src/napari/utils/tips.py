import re
import warnings

from napari.utils.interactions import Shortcut

NAPARI_TIPS: tuple[str, ...] = (
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


def format_tip(tip: str) -> str:
    """Format a tip with appropriate shortcuts and command keybindings.

    Input tip should use template string formatting.
    Expands to platform-specific glyphs as appropriate.
    Accepts direct shortcuts such as `{Alt+X}` and command ids
    such as `napari.window.file.copy_canvas_screenshot`.

    Note: for some actions, the napari viewer needs to be initialized once
    in order for them to be registered.
    """
    from napari._app_model.utils import get_command_shortcut_and_description

    # TODO: this should use template strings in the future
    for match in re.finditer(r'{(.*?)}', tip):
        command_id = match.group(1)
        shortcut, _ = get_command_shortcut_and_description(command_id)
        # this can be none at launch (not yet initialized), will be updated after
        if shortcut is None:
            # maybe it was just a direct keybinding given
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                shortcut = Shortcut(command_id).platform
        if shortcut:
            tip = re.sub(match.group(), str(shortcut), tip)
    return tip
