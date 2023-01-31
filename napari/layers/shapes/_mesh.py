import numpy as np


class Mesh:
    """Contains meshses of shapes that will ultimately get rendered.

    Parameters
    ----------
    ndisplay : int
        Number of displayed dimensions.

    Attributes
    ----------
    ndisplay : int
        Number of displayed dimensions.
    vertices : np.ndarray
         Qx2 array of vertices of all triangles for shapes including edges and
         faces
    vertices_centers : np.ndarray
         Qx2 array of centers of vertices of triangles for shapes. For vertices
         corresponding to faces these are the same as the actual vertices. For
         vertices corresponding to edges these values should be added to a
         scaled `vertices_offsets` to get the actual vertex positions.
         The scaling corresponds to the width of the edge
    vertices_offsets : np.ndarray
         Qx2 array of offsets of vertices of triangles for shapes. For vertices
         corresponding to faces these are 0. For vertices corresponding to
         edges these values should be scaled and added to the
         `vertices_centers` to get the actual vertex positions.
         The scaling corresponds to the width of the edge
    vertices_index : np.ndarray
         Qx2 array of the index (0, ..., N-1) of each shape that each vertex
         corresponds and the mesh type (0, 1) for face or edge.
    triangles : np.ndarray
        Px3 array of vertex indices that form the mesh triangles
    triangles_index : np.ndarray
        Px2 array of  the index (0, ..., N-1) of each shape that each triangle
        corresponds and the mesh type (0, 1) for face or edge.
    triangles_colors : np.ndarray
        Px4 array of the rgba color of each triangle
    triangles_z_order : np.ndarray
        Length P array of the z order of each triangle. Must be a permutation
        of (0, ..., P-1)

    Notes
    -----
    _types : list
        Length two list of the different mesh types corresponding to faces and
        edges
    """

    _types = ['face', 'edge']

    def __init__(self, ndisplay=2) -> None:

        self._ndisplay = ndisplay
        self.clear()

    def clear(self):
        """Resets mesh data"""
        self.vertices = np.empty((0, self.ndisplay))
        self.vertices_centers = np.empty((0, self.ndisplay))
        self.vertices_offsets = np.empty((0, self.ndisplay))
        self.vertices_index = np.empty((0, 2), dtype=int)
        self.triangles = np.empty((0, 3), dtype=np.uint32)
        self.triangles_index = np.empty((0, 2), dtype=int)
        self.triangles_colors = np.empty((0, 4))
        self.triangles_z_order = np.empty((0), dtype=int)

        self.displayed_triangles = np.empty((0, 3), dtype=np.uint32)
        self.displayed_triangles_index = np.empty((0, 2), dtype=int)
        self.displayed_triangles_colors = np.empty((0, 4))

    @property
    def ndisplay(self):
        """int: Number of displayed dimensions."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay):
        if self.ndisplay == ndisplay:
            return

        self._ndisplay = ndisplay
        self.clear()
