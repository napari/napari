"""
Display a 3D sphere with normals and wireframe
"""

try:
    from meshzoo import icosa_sphere
except ImportError as e:
    raise ImportError(
        "This example uses a meshzoo but meshzoo is not installed. "
        "To install try 'pip install meshzoo'."
    ) from e
import napari


vert, faces = icosa_sphere(10)

vert *= 100

viewer = napari.Viewer(ndisplay=3)
surface1 = viewer.add_surface(
    data=(vert, faces),
    wireframe=True,
    face_normals=True,
    face_normals_length=-10,  # negative length cause meshzoo has inverted normals
    face_normals_color='yellow',
    vertex_normals=True,
    vertex_normals_length=-10,
    vertex_normals_color='blue',
)
viewer.reset_view()

napari.run()
