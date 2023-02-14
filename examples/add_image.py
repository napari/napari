"""
Add image
=========

Display one image using the :func:`view_image` API.

.. tags:: visualization-basic
"""

from skimage import data

import napari

# create the viewer with an image
viewer = napari.view_image(data.astronaut(), rgb=True)

print(f'system theme: {napari.utils.theme.get_system_theme()}')

if __name__ == '__main__':
    napari.run()
