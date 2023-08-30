"""
Surface normals wireframe
=========================

Display a 3D mesh with normals and wireframe

.. tags:: experimental
"""

from vispy.io import load_data_file, read_mesh

import napari

vert, faces, _, _ = read_mesh(load_data_file('orig/triceratops.obj.gz'))

# put the mesh right side up, scale it up (napari#3477) and fix faces handedness
vert *= -100
faces = faces[:, ::-1]

viewer = napari.Viewer(ndisplay=3)
surface = viewer.add_surface(data=(vert, faces))

# enable normals and wireframe
surface.normals.face.visible = True
surface.normals.vertex.visible = True
surface.wireframe.visible = True

if __name__ == '__main__':
    napari.run()
