"""
New theme
=========

Displays an image and sets the theme to new custom theme.

.. tags:: experimental
"""

from skimage import data
import napari
from napari.utils.theme import available_themes, get_theme, register_theme


# create the viewer with an image
viewer = napari.view_image(data.astronaut(), rgb=True, name='astronaut')

# List themes
print('Originally themes', available_themes())

blue_theme = get_theme('dark', False)
blue_theme.name = "blue"
blue_theme.icon = (
    'rgb(0, 255, 255)'  # you can provide colors as rgb(XXX, YYY, ZZZ)
)
blue_theme.background = 28, 31, 48  # or as tuples
blue_theme.foreground = [45, 52, 71]  # or as list
blue_theme.primary = '#50586c'  # or as hexes
blue_theme.current = 'orange'  # or as color name

register_theme('blue', blue_theme)

# List themes
print('New themes', available_themes())

# Set theme
viewer.theme = 'blue'

if __name__ == '__main__':
    napari.run()
