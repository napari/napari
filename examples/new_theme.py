"""
Displays an image and sets the theme to new custom theme.
"""

from skimage import data
import napari
from napari.utils.theme import available_themes, get_theme, register_theme

with napari.gui_qt():
    # create the viewer with an image
    viewer = napari.view_image(
        data.astronaut(), rgb=True, name='astronaut'
    )

    # List themes
    print('Originally themes', available_themes())

    blue_theme = get_theme('dark')
    blue_theme.update(
        background='rgb(28, 31, 48)',
        foreground='rgb(45, 52, 71)',
        primary='rgb(80, 88, 108)',
        current='rgb(184, 112, 0)',
    )

    register_theme('blue', blue_theme)

    # List themes
    print('New themes', available_themes())

    # Set theme
    viewer.theme = 'blue'
