"""
Displays an image and sets the theme to new custom theme.
"""

from skimage import data
import napari
from napari.utils.theme import palettes

with napari.gui_qt():
    # create the viewer with an image
    viewer = napari.view_image(
        data.astronaut(), rgb=True, name='astronaut'
    )

    # Create a new palette. Note we must use either the `light` or
    # `dark` folder to avoid needing to build new icons
    blue_palette = palettes['dark'].copy()
    blue_palette['background'] = 'rgb(28, 31, 48)'
    blue_palette['foreground'] = 'rgb(45, 52, 71)'
    blue_palette['primary'] = 'rgb(80, 88, 108)'
    blue_palette['current'] = 'rgb(184, 112, 0)'

    # Add new theme to dictionary of available_themes
    palettes['blues'] = blue_palette

    # Set theme
    viewer.theme = 'blues'
