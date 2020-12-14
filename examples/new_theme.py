"""
Displays an image and sets the theme to 'light'.
"""

from skimage import data
import napari
from napari.utils.theme import Palette, available_themes

with napari.gui_qt():
    # create the viewer with an image
    viewer = napari.view_image(
        data.astronaut(), rgb=True, name='astronaut'
    )

    # Create a new palette. Note we must use either the `light` or
    # `dark` folder to avoid needing to build new icons
    blue_palette = Palette(
        folder='dark',
        background='rgb(28, 31, 48)',
        foreground='rgb(45, 52, 71)',
        primary='rgb(80, 88, 108)',
        secondary='rgb(134, 142, 147)',
        highlight='rgb(106, 115, 128)',
        text='rgb(240, 241, 242)',
        icon='rgb(209, 210, 212)',
        warning='rgb(153, 18, 31)',
        current='rgb(184, 112, 0)',
        syntax_style='native',
        console='rgb(0, 0, 0)',
        canvas='black',
    )
    # Add new theme to dictionary of available_themes
    available_themes['blues'] = blue_palette

    # Set theme
    viewer.theme = 'blues'
