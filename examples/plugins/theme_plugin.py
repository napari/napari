"""Very simple theme plugin example providing simple color scheme changes."""
from napari_plugin_engine import napari_hook_implementation


def custom_qss():
    """Get list qss stylesheets."""
    qss_files = ["04_theme.qss"]
    return qss_files


def custom_icons():
    """Get list of svg icons."""
    svg_icons = ["delete.svg"]
    return svg_icons


def custom_theme():
    """Get dictionary of color schemes."""
    themes = {
        "super_dark": {
            "folder": "super_dark",
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
def napari_experimental_provide_qss():
    """A basic implementation of the `napari_experimental_provide_qss` hook specification."""
    return custom_qss


@napari_hook_implementation
def napari_experimental_provide_icons():
    """A basic implementation of the `napari_experimental_provide_icons` hook specification."""
    return custom_icons


@napari_hook_implementation
def napari_experimental_provide_theme():
    """A basic implementation of the `napari_experimental_provide_theme` hook specification."""
    return custom_theme()
