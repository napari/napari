import numpy as np

from napari.layers.shapes.shape_types import (
    CoordinateArray,
    CoordinateDtype,
    IndexArray,
    IndexDtype,
    ShapeColorArray,
    ShapeColorDtype,
    TriangleArray,
    TriangleDtype,
    ZOrderArray,
    ZOrderDtype,
)


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
         faces For each shape, we store the vertices of the triangles first, then vertices of edges
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

    _types = ('face', 'edge')
    vertices: CoordinateArray
    vertices_centers: CoordinateArray
    vertices_offsets: CoordinateArray
    triangles: TriangleArray
    triangles_z_order: ZOrderArray
    triangles_colors: ShapeColorArray
    displayed_triangles: TriangleArray
    displayed_triangles_colors: ShapeColorArray
    displayed_triangles_to_shape_index: IndexArray
    vertices_index: IndexArray
    triangles_index: IndexArray

    def __init__(self, ndisplay: int = 2) -> None:
        self._ndisplay = ndisplay
        self.clear()

    def clear(self) -> None:
        """Resets mesh data"""
        self.vertices = np.empty((0, self.ndisplay), dtype=CoordinateDtype)  # type: ignore[assignment]
        self.vertices_centers = np.empty(  # type: ignore[assignment]
            (0, self.ndisplay), dtype=CoordinateDtype
        )
        self.vertices_offsets = np.empty(  # type: ignore[assignment]
            (0, self.ndisplay), dtype=CoordinateDtype
        )
        self.vertices_index = np.zeros(1, dtype=IndexDtype)
        self.triangles = np.empty((0, 3), dtype=TriangleDtype)  # type: ignore[assignment]
        self.triangles_index = np.zeros(1, dtype=IndexDtype)
        self.triangles_colors = np.empty((0, 4), dtype=ShapeColorDtype)  # type: ignore[assignment]
        self.triangles_z_order = np.empty(0, dtype=ZOrderDtype)

        self.displayed_triangles = np.empty((0, 3), dtype=TriangleDtype)  # type: ignore[assignment]
        self.displayed_triangles_to_shape_index = np.zeros(1, dtype=IndexDtype)
        self.displayed_triangles_colors = np.empty(  # type: ignore[assignment]
            (0, 4), dtype=ShapeColorDtype
        )

    @property
    def ndisplay(self) -> int:
        """int: Number of displayed dimensions."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay: int) -> None:
        if self.ndisplay == ndisplay:
            return

        self._ndisplay = ndisplay
        self.clear()
