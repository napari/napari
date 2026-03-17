from typing import Literal

from napari import __version__
from napari.components.overlays.base import CanvasOverlay
from napari.utils.tips import NAPARI_TIPS


class WelcomeOverlay(CanvasOverlay):
    """Welcome screen overlay."""

    # not settable in this specific overlay
    position: None = None
    # ensure it's on top of overlays with default value
    order: int = 10**6 + 10
    gridded: Literal[False] = False
    version: str = __version__
    shortcuts: tuple[str, ...] = (
        'napari.window.file._image_from_clipboard',
        'napari.window.file.open_files_dialog',
        'napari.window.view.toggle_command_palette',
        'napari:show_shortcuts',
    )
    # TODO: query mouse binding as well somehow? Currently we have to hardcode those.
    tips: tuple[str, ...] = NAPARI_TIPS
