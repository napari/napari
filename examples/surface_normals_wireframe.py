"""
Display a 3D mesh with normals and wireframe
"""

from vispy.io import read_mesh, load_data_file
import napari


vert, faces, _, _ = read_mesh(load_data_file('orig/triceratops.obj.gz'))

vert *= 100  # c.f. https://github.com/napari/napari/issues/3477

viewer = napari.Viewer(ndisplay=3)
surface = viewer.add_surface(
    data=(vert, faces),
    wireframe=True,
    face_normals=True,
    vertex_normals=True,
)
viewer.reset_view()

napari.run()
