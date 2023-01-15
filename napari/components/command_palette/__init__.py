from napari.components.command_palette._api import CommandPalette, get_palette
from napari.components.command_palette._components import Command

__all__ = [
    "Command",
    "get_palette",
    "CommandPalette",
    "get_napari_command_palette",
]


_PALETTE = None


def get_napari_command_palette() -> CommandPalette:
    global _PALETTE

    if _PALETTE is not None:
        return _PALETTE

    from napari._app_model import get_app

    app = get_app()
    palette = get_palette(app.name)

    for name, cmd in app.commands:
        *contexts, _ = name.split(":")
        title = " > ".join(contexts)
        desc = cmd.title
        func = app.injection_store.inject(cmd.resolved_callback)
        palette.register(
            func,
            title=title,
            desc=desc,
        )

    _PALETTE = palette

    return palette
