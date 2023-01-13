"""
Spheres
=======

Display two spheres with Surface layers

.. tags:: visualization-advanced
"""

import napari
from vispy.geometry import create_sphere

mesh = create_sphere(method='ico')

faces = mesh.get_faces()
vert = mesh.get_vertices() * 100

sphere1 = (vert + 30, faces)
sphere2 = (vert - 30, faces)

viewer = napari.Viewer(ndisplay=3)
surface1 = viewer.add_surface(sphere1)
surface2 = viewer.add_surface(sphere2)
viewer.reset_view()

if __name__ == '__main__':
    napari.run()
