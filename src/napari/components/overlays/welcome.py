from typing import Literal

from napari import __version__
from napari.components.overlays.base import CanvasOverlay
from napari.utils.color import ColorValue


class WelcomeOverlay(CanvasOverlay):
    """Welcome screen overlay."""

    # not settable in this specific overlay
    position: None = None
    # ensure it's on top of overlays with default value
    order: int = 10**6 + 1
    gridded: Literal[False] = False
    version: str = __version__
    shortcuts: tuple[str, ...] = (
        'napari.window.file._image_from_clipboard',
        'napari.window.file.open_files_dialog',
        'napari.window.view.toggle_command_palette',
        'napari:show_shortcuts',
    )
    # TODO: query mouse binding as well somehow? Currently we have to hardcode those.
    tips: tuple = (
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
