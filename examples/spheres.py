"""
Display two spheres with Surface layers
"""

import napari
from meshzoo import icosa_sphere


vert, faces = icosa_sphere(10)

vert *= 100

sphere1 = (vert + 30, faces)
sphere2 = (vert - 30, faces)

viewer = napari.Viewer(ndisplay=3)
surface1 = viewer.add_surface(sphere1)
surface2 = viewer.add_surface(sphere2)
viewer.reset_view()

napari.run()
