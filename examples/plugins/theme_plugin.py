"""Very simple theme plugin example providing simple color scheme changes."""
from napari_plugin_engine import napari_hook_implementation


def custom_theme():
    """Get dictionary of color schemes."""
    themes = {
        "super_dark": {
            "name": "super_dark",
            "background": "rgb(12, 12, 12)",
            "foreground": "rgb(65, 72, 81)",
            "primary": "rgb(90, 98, 108)",
            "secondary": "rgb(134, 142, 147)",
            "highlight": "rgb(106, 115, 128)",
            "text": "rgb(240, 241, 242)",
            "icon": "rgb(209, 210, 212)",
            "warning": "rgb(153, 18, 31)",
            "current": "rgb(0, 122, 204)",
            "syntax_style": "native",
            "console": "rgb(0, 0, 0)",
            "canvas": "black",
        }
    }
    return themes


@napari_hook_implementation
def napari_provide_theme():
    """A basic implementation of the `napari_provide_theme` hook specification."""
    return custom_theme()
