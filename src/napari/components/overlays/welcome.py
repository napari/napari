from typing import Literal

from napari import __version__
from napari.components.overlays.base import CanvasOverlay


class WelcomeOverlay(CanvasOverlay):
    """Welcome screen overlay."""

    visible: bool = True
    # not settable in this specific overlay
    position: None = None
    gridded: Literal[False] = False
    version: str = __version__
    shortcuts: tuple[str, ...] = (
        'napari.window.file._image_from_clipboard',
        'napari.window.file.open_files_dialog',
        'napari.window.view.toggle_command_palette',
        'napari:show_shortcuts',
    )
    tip: str = ''
